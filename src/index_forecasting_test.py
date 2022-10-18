from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""
import header.index_forecasting.RUNHEADER as RUNHEADER
import util

# from multiprocessing.managers import BaseManager
import numpy as np

# import pandas as pd

# from datasets.index_forecasting_protobuf2pickle import DataSet
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
# from sklearn.metrics import mean_squared_error
import os
import pickle

# import sys
# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.use('agg')
# import shutil
# import argparse
# from mpl_finance import candlestick_ohlc
# from matplotlib.dates import date2num
# import datetime
import cloudpickle
import plot_util

# import copy


class Script:
    def __init__(self, so=None):
        self.so = so
        self.base_model = None

    def run(
        self,
        mode,
        env_name=None,
        tensorboard_log=None,
        full_tensorboard_log=None,
        model_location=None,
        n_cpu=None,
        n_step=None,
        total_timesteps=None,
        result=None,
        m_inference_buffer=None,
        b_naive=True,
    ):

        # get model list for evaluate performance
        models = os.listdir(model_location)

        # Todo: Distributed inference code
        self._inference(
            models, env_name, n_cpu, mode, model_location, result, m_inference_buffer, b_naive
        )

    def _inference(
        self, models, env_name, n_cpu, mode, model_location, result, m_inference_buffer, b_naive=True
    ):
        # import modules
        from custom_model.index_forecasting.common import SubprocVecEnv
        from custom_model.index_forecasting import A2C

        dump_header = convert_pickable(RUNHEADER)
        env = SubprocVecEnv(
            [[lambda: util.make(env_name), dump_header] for i in range(n_cpu)]
        )  # fork running RUNHEADER.py version

        # env = SubprocVecEnv([lambda: util.make(env_name) for i in range(n_cpu)])
        env.set_attr("so", self.so)
        env.set_attr("mode", mode)

        filenames = list()
        if (RUNHEADER.m_final_model is None) or (RUNHEADER.m_final_model is "None"):
            [filenames.append(_model) for _model in models if ".pkl" in _model]
            filenames.sort()
        else:
            [
                filenames.append(_model)
                for _model in models
                if RUNHEADER.m_final_model in _model
            ]

        # naive filter for a model
        if b_naive:
            filenames = naive_filter(filenames)
        is_graph_def_loaded = False
        for _model in filenames:
            """Inference"""
            current_step = 0
            env.set_attr("current_step", current_step)
            env.set_attr("eof", False)
            env.set_attr("steps_beyond_done", None)
            env.env_method("clear_cache")
            env.env_method("clear_cache_test_step")
            obs = env.reset()

            _model = "{}/{}".format(model_location, _model)
            print("\n\nloading model: {} ".format(_model))

            if (
                not is_graph_def_loaded
            ):  # reset graph every 10times, to avoid memory overflow - fixed??? check
                # self.base_model = A2C.inf_init(_model, policy_kwargs={'n_lstm': 256 * RUNHEADER.m_lstm_hidden,
                #                                                                 'is_training': False})
                self.base_model = A2C.inf_init(
                    _model,
                    env=env,
                    policy_kwargs={
                        "n_lstm": int(256 * RUNHEADER.m_lstm_hidden),
                        "is_training": False,
                    },
                )
                is_graph_def_loaded = True
            self.params_load(_model)

            tmp_info, selected_fund, img_action, act_selection = (
                list(),
                list(),
                list(),
                list(),
            )
            p_states, action, info = None, None, None
            while np.sum(np.array(env.get_attr("eof"))) == 0:
                """Inference with continuous state matrix"""
                # # Original Version
                # action, states, values, neglogp, values2 = self.base_model.predict(
                #     obs, state=p_states, mask=None, deterministic=True
                # )

                # Fast approach
                action, states, values, neglogp, values2 = self.base_model.predict(
                    obs + np.zeros([RUNHEADER.m_n_cpu, obs.shape[1], obs.shape[2], obs.shape[3]]), state=p_states, mask=None, deterministic=True
                )

                # disable
                # action_pro, softmax_actions = self.base_model.action_probability(obs, state=p_states, mask=None,
                #                                                        actions=action)
                softmax_actions = np.zeros((1, 5, 2))  # a dummy code for fast inference

                p_states = states
                _, rewards, _, info = env.step(action)

                """Bound Estimation
                """
                b_info = list()
                b_info_return = list()
                _info = None
                if (
                    current_step >= m_inference_buffer
                ) and RUNHEADER.m_bound_estimation:
                    for sub_current_step in range(m_inference_buffer):
                        if sub_current_step > 0:
                            states = None
                            _values = None
                            _action = np.ones(
                                (env.num_envs, env.action_space.shape[0]),
                                dtype=np.ndarray,
                            )
                            for inner_step in range(sub_current_step - 1, -1, -1):
                                env.set_attr("current_step", current_step - inner_step)
                                obs, _, _, _ = env.step(_action)
                                (
                                    _action,
                                    states,
                                    _values,
                                    _,
                                    _values2,
                                ) = self.base_model.predict(
                                    obs, state=states, mask=None, deterministic=True
                                )
                            _, _, _, _info = env.step(_action)
                            b_info.append(
                                ((_info[0]["today_index"] * _values[0]) / 100)
                                + _info[0]["today_index"]
                            )
                            b_info_return.append(_values[0])
                    index_bound = [
                        np.min(np.array(b_info)),
                        np.max(np.array(b_info)),
                        np.mean(np.array(b_info)),
                        np.std(np.array(b_info)),
                        np.median(np.array(b_info)),
                    ]
                    index_bound_return = [
                        np.min(np.array(b_info_return)),
                        np.max(np.array(b_info_return)),
                        np.mean(np.array(b_info_return)),
                        np.std(np.array(b_info_return)),
                        np.median(np.array(b_info_return)),
                    ]
                else:
                    # dummy values for plotting
                    index_bound = [0, 0, 0, 0, 0]
                    index_bound_return = [0, 0, 0, 0, 0]

                """ Next Step and performance calculation for current step
                """
                current_step = current_step + 1
                env.set_attr("current_step", current_step)
                obs, _, _, _ = env.step(action)

                # gather performance
                if RUNHEADER.m_warm_up_4_inference < current_step:
                    tmp_info = plot_util.gather_validation_performence(
                        info,
                        tmp_info,
                        values,
                        values2,
                        softmax_actions,
                        index_bound,
                        index_bound_return,
                        b_info,
                        b_info_return,
                    )

            """File out information
            """
            plot_util.plot_save_validation_performence(
                tmp_info, result, _model, split_name=mode
            )

        # File out Summary by epoch
        plot_util.epoch_summary(result)

        # # # adhoc-process
        # plot_util.adhoc_process(filenames, result)

    def load_from_file(self, load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError(
                        "Error: the file {} could not be found".format(load_path)
                    )
            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    def params_load(self, load_path):
        _, params = self.load_from_file(load_path)
        self.base_model.weight_load_ph(params)

    def learning_parameter(self, result):
        keys = [
            key
            for key in RUNHEADER.__dict__.keys()
            if type(RUNHEADER.__dict__[key]) in [int, str, float, bool]
        ]

        # Export to txt
        file_location = "{}/fig_index/".format(result)
        with open(file_location + "agent_parameter.txt", "w") as f_out:
            for element in keys:
                print(
                    "{} : {}".format(element, RUNHEADER.__dict__[element]), file=f_out
                )
        f_out.close()


def recent_procedure(file_name, process_id, mode):
    json_file_location = ""
    with open("{}{}.txt".format(file_name, str(process_id)), mode) as _f_out:
        if mode == "w":
            print(RUNHEADER.m_name, file=_f_out)
        elif mode == "r":
            json_file_location = _f_out.readline()
        else:
            assert False, "<recent_procedure> : mode error"
        _f_out.close()
    return json_file_location.replace("\n", "")


def get_keys(module_name):
    general_type = [
        key
        for key in module_name.__dict__.keys()
        if type(module_name.__dict__[key]) in [int, str, float, bool]
    ]
    none_type = [
        key for key in module_name.__dict__.keys() if module_name.__dict__[key] is None
    ]
    return general_type + none_type


def convert_pickable(module_name):
    keys = get_keys(module_name)
    _dict = [[element, RUNHEADER.__dict__[element]] for element in keys]
    return dict(_dict)


def init_start(header):
    for key in header.keys():
        RUNHEADER.__dict__[key] = header[key]


def load(filepath, method):
    with open(filepath, "rb") as fs:
        if method == "pickle":
            data = pickle.load(fs)
    fs.close()
    return data
 

def get_model_from_meta_repo(target_name, forward, use_historical_model=False):
    a, b, c, d = list(), list(), list(), list()
    model_info = load(
        "./save/model_repo_meta/{}_T{}.pkl".format(target_name, forward), "pickle"
    )
    for model in model_info:
        if not use_historical_model:
            if model["latest"]:  # the best at the moment
                return model["m_name"], model["model_name"]
        else:
            if os.path.isfile('./save/model/rllearn/{}/{}'.format(model["m_name"], model["model_name"])):
                a.append(model["m_name"])
                b.append(model["model_name"])
                # use current best, if the model exist for the current_period
                if model["current_period"]:  
                    c.append(True)
                else:  # use hiatorical best
                    c.append(False)
                d.append(model["m_offline_buffer_file"])

    if len(a) > 0 and len(b) > 0:
        e = np.array(list(set(list(zip(a, b, c, d)))))
        return e[:, 0].tolist(), e[:, 1].tolist(), e[:, 2].tolist(), e[:, 3].tolist()
    assert False, "There are no a latest tagged model"


def configure_header(args):
    json_location = recent_procedure("./agent_log/working_model_p", args.process_id, "r")
    # keep explcit test model before re-load RUNHEADER
    f_test_model = RUNHEADER.m_final_model
    dict_RUNHEADER = util.json2dict(
        "./save/model/rllearn/{}/agent_parameter.json".format(json_location)
    )

    # re-load
    for key in dict_RUNHEADER.keys():
        if (key == "_debug_on") or (key == "release") or (key == "c_epoch"):
            pass  # use global parameter
        else:
            RUNHEADER.__dict__[key] = dict_RUNHEADER[key]
    RUNHEADER.__dict__["m_final_model"] = f_test_model
    RUNHEADER.__dict__["m_bound_estimation"] = False
    RUNHEADER.__dict__["m_bound_estimation_y"] = True
    # RUNHEADER.__dict__["m_warm_up_4_inference"] = RUNHEADER.forward_ndx
    # RUNHEADER.__dict__["m_warm_up_4_inference"] = 6
    

def meta_info(_model_location, _dataset_dir):
    # load meta from trained model
    with open(_model_location + "/meta", mode="rb") as fp:
        meta_1 = pickle.load(fp)
        fp.close()
    # # load meta from dataset version - Disable for operation mode
    # with open(_dataset_dir + "/meta", mode="rb") as fp:
    #     meta_2 = pickle.load(fp)
    #     fp.close()

    # basic info
    n_step = meta_1["_n_step"]
    cv_number = meta_1["_cv_number"]
    n_cpu = meta_1["_n_cpu"]
    env_name = meta_1["_env_name"]
    file_pattern = meta_1["_file_pattern"]

    # # define the data set for a inference - Disable for operation mode
    # infer_set = meta_2[
    #     "verbose"
    # ]  # infer_set contains one of [train | validation | test]
    # if meta_2["verbose"] == 2:  # TRAIN_WITHOUT_VAL
    #     infer_set = ["test"]
    # elif meta_2["verbose"] == 3 or meta_2["verbose"] == 4:
    #     infer_set = ["test", "validation"]  # TRAIN_WITH_VAL_D
    # else:
    #     assert False, "Not defined yet!!!"
    infer_set = ["test", "validation"]

    return n_step, cv_number, n_cpu, env_name, file_pattern, infer_set


def naive_filter(model_list):
    c_pe, c_pl, c_vl, c_ev, c_epoch = 3, 3, 2, 0.75, RUNHEADER.c_epoch
    filtered_model = list()
    for model_name in model_list:
        token = model_name.split("_")
        try:
            if (
                float(token[5][2:]) < c_pe
                and float(token[6][2:]) < c_pl
                and float(token[7][2:]) < c_vl
                and float(token[8][2:-4]) > c_ev
                and int(token[4]) >= c_epoch
            ):
                filtered_model.append(model_name)
        except IndexError:
            filtered_model.append(model_name)
    assert len(filtered_model), "All model are denied from the naive filter"

    return sorted(list(set(filtered_model)), key=len)
