# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""
from __future__ import absolute_import, division, print_function

import datetime
import pickle
from typing import Any, Dict, List, Tuple

import header.index_forecasting.RUNHEADER as RUNHEADER
import util

# from multiprocessing.managers import BaseManager
# import shutil
# import os
# import argparse
# from datasets.index_forecasting_protobuf2pickle import DataSet


class Script:
    def __init__(self, so=None, so_validation=None) -> None:
        self.so: Any = so
        self.so_validation: Any = so_validation

    def run(
        self,
        mode,
        env_name=None,
        tensorboard_log=None,
        verbose=None,
        full_tensorboard_log=None,
        model_location=None,
        learning_rate=None,
        n_cpu=None,
        n_step=None,
        total_timesteps=None,
        log_interval=None,
    ) -> None:

        # save learning parameters
        self.learning_parameter(model_location)

        # import modules
        dump_header = convert_pickable(RUNHEADER)
        from custom_model.index_forecasting.a2c import A2C
        from custom_model.index_forecasting.common import SubprocVecEnv
        from custom_model.index_forecasting.policies.policies_index_forecasting import (
            CnnLnLstmPolicy,
        )

        # generate environments
        if RUNHEADER.m_online_buffer == 1:
            env: SubprocVecEnv = SubprocVecEnv(
                [[lambda: util.make(env_name), dump_header] for i in range(n_cpu)]
            )  # fork running RUNHEADER.py version
        else:
            env: SubprocVecEnv = SubprocVecEnv(
                [[lambda: util.make(env_name), dump_header] for i in range(1)]
            )  # fork running RUNHEADER.py version

        # init environments
        env.set_attr("so", self.so)
        env.set_attr("so_validation", self.so_validation)
        env.set_attr("mode", mode)
        env.set_attr("current_episode_idx", 0)
        env.set_attr("current_step", 0)
        if RUNHEADER.m_train_mode == 0:  # call network graph
            model: A2C = A2C(
                CnnLnLstmPolicy,
                env,
                verbose=verbose,
                n_steps=n_step,
                learning_rate=learning_rate,
                tensorboard_log=tensorboard_log,
                full_tensorboard_log=full_tensorboard_log,
                policy_kwargs={
                    "n_lstm": int(256 * RUNHEADER.m_lstm_hidden),
                    "is_training": True,
                },
            )
        else:  # use pre-trained model
            print("\nloading model ")
            model: A2C = A2C.load(RUNHEADER.m_pre_train_model, env)

        # seed=0 fix seed
        model.learn(
            total_timesteps=total_timesteps,
            seed=0,
            model_location=model_location,
            log_interval=log_interval,
        )
        model.save(model_location)
        del model  # remove to demonstrate saving and loading

    def learning_parameter(self, model_location) -> None:
        keys = get_keys(RUNHEADER)

        # Export to txt
        with open(model_location + "/agent_parameter.txt", "w") as f_out:
            for element in keys:
                print(f"{element} : {RUNHEADER.__dict__[element]}", file=f_out)
        f_out.close()

        # Save to json
        util.dict2json(
            model_location + "/agent_parameter.json",
            {element: RUNHEADER.__dict__[element] for element in keys},
        )


def recent_procedure(file_name, process_id, mode):
    json_file_location = ""

    with open(f"{file_name}{str(process_id)}.txt", mode) as _f_out:
        if mode == "w":
            print(RUNHEADER.m_name, file=_f_out)
        elif mode == "r":
            json_file_location = _f_out.readline()
        else:
            assert False, "<recent_procedure> : mode error"
        _f_out.close()
    return json_file_location.replace("\n", "")


def configure_header(args):
    time_now: str = (
        str(datetime.datetime.now())[:-10]
        .replace(":", "-")
        .replace("-", "")
        .replace(" ", "_")
    )
    selected_x_dict: Dict = {}
    json_location: str = ""

    # set from arguments parser
    RUNHEADER.__dict__["m_online_buffer"] = args.m_online_buffer
    RUNHEADER.__dict__["search_variables"] = args.search_variables
    RUNHEADER.__dict__["search_parameter"] = args.search_parameter

    if RUNHEADER.m_online_buffer:  # Generate Buffer
        assert (
            bool(args.on_cloud) is False
        ), "on_cloud should be false for generate Buffer"

        RUNHEADER.__dict__["forward_ndx"] = args.forward_ndx
        RUNHEADER.__dict__[
            "m_dataset_dir"
        ] = f"./save/tf_record/{str(RUNHEADER.tf_record_location)}/{str(RUNHEADER.l_objective)}_x0_20_y{str(RUNHEADER.forward_ndx)}_{args.dataset_version}"
        RUNHEADER.__dict__["m_total_example"] = 0
        with open(RUNHEADER.m_dataset_dir + "/meta", mode="rb") as fp_meta:
            info: Any = pickle.load(fp_meta)
            fp_meta.close()
        assert (RUNHEADER.__dict__["forward_ndx"]) == (
            info["forecast"]
        ), "Wrong DataSet is selected for a forward_ndx forecast"

        if args.search_variables:  # random pick x variables
            RUNHEADER.__dict__["m_name"] = (
                RUNHEADER.m_name + "_T" + str(RUNHEADER.forward_ndx) + "_" + time_now
            )
            selected_x_dict = util.json2dict(RUNHEADER.m_dataset_dir + "/x_index.json")
            # modify code .. Add random pick, later on
            # selected_x_dict = selected_x_dict  # Add Logic
        else:
            RUNHEADER.__dict__["m_name"] = (
                RUNHEADER.m_name + "_T" + str(RUNHEADER.forward_ndx) + "_" + time_now
            )
        RUNHEADER.__dict__["m_offline_buffer_file"] = (
            "./save/model/rllearn/buffer_save/" + RUNHEADER.m_name
        )
        RUNHEADER.__dict__["m_on_validation"] = False
        RUNHEADER.__dict__[
            "weighted_random_sample"
        ] = True  # last 2 month (3 times over sampling)

        # following are hard coded
        RUNHEADER.__dict__["m_lstm_hidden"] = RUNHEADER.m_lstm_hidden
        RUNHEADER.__dict__["m_num_features"] = RUNHEADER.m_num_features
        # RUNHEADER.__dict__['m_n_cpu'] = RUNHEADER.m_n_cpu
        RUNHEADER.__dict__["m_n_step"] = RUNHEADER.m_n_step
        tmp = RUNHEADER.m_n_step
        # RUNHEADER.__dict__['dataset_version'] = 'v8_1'
        RUNHEADER.__dict__["dataset_version"] = args.dataset_version
        RUNHEADER.__dict__["m_n_cpu"] = args.n_cpu  # populate rate
        RUNHEADER.__dict__[
            "enable_lstm"
        ] = False  # when continuous learning, the parameter should be configured
        RUNHEADER.__dict__["re_assign_vars"] = False
        assert tmp == RUNHEADER.m_n_step, "check dataset version"

        recent_procedure("./agent_log/buffer_generate_model_p", args.process_id, "w")

    else:  # Learning with Buffer
        RUNHEADER.__dict__[
            "dataset_version"
        ] = None  # only use for checking training performance

        json_location = recent_procedure(
            "./agent_log/buffer_generate_model_p", args.process_id, "r"
        )
        dict_RUNHEADER: Any = util.json2dict(
            f"./save/model/rllearn/{json_location}/agent_parameter.json"
        )
        # load RUNHEADER
        for key in dict_RUNHEADER.keys():
            if (key == "_debug_on") or (key == "release") or (key == "c_epoch"):
                pass  # use global parameter
            else:
                RUNHEADER.__dict__[key] = dict_RUNHEADER[key]
        # re-write
        RUNHEADER.__dict__["m_online_buffer"] = args.m_online_buffer
        RUNHEADER.__dict__["search_variables"] = args.search_variables
        RUNHEADER.__dict__["search_parameter"] = args.search_parameter
        RUNHEADER.__dict__["m_n_cpu"] = args.n_cpu  # agent number
        # RUNHEADER.__dict__['n_off_batch'] = args.n_off_batch  # agent number
        RUNHEADER.__dict__["m_train_mode"] = 0
        # sub1, 140 -> sub2, 70 -> sub2, 140 -> sub1, 200
        RUNHEADER.__dict__[
            "m_offline_learning_epoch"
        ] = 50  # total epochs should reach to about 140 epochs (experimental result) 300 -> 120 (For fast experimental) -> 150 -> 50
        RUNHEADER.__dict__[
            "m_sub_epoch"
        ] = 1  # investigate the same samples 3 times in a epoch to increase training speed
        RUNHEADER.__dict__[
            "on_cloud"
        ] = args.on_cloud  # load whole samples on the memory
        RUNHEADER.__dict__["enable_lstm"] = True

        # remove it
        # if args.on_cloud:
        #     RUNHEADER.__dict__['m_buffer_size'] = int(
        #         RUNHEADER.m_total_example * RUNHEADER.m_n_cpu * RUNHEADER.m_n_step * 0.5)

        # Specifiy parameter: Add more code for search
        if args.search_parameter == 0:  # fix fix fix
            RUNHEADER.__dict__[
                "default_net"
            ] = "inception_resnet_v2_D"  # Slow Convergence than inception_resnet_v2_G, check performence
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 200

            RUNHEADER.__dict__["m_on_validation"] = False
            # True -> False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [5e-4, 5e-4, 5e-4]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 5e-4  # a made decision
            #  5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 5e-4  # a made decision 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 1e-5  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 0.2  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 0.2  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__["m_pi_coef"] = 1  # 0.01 -> 1 -> 0.05 -> 0.1
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m1_0_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 1:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 2e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 6e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 1  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 1  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 0.7  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_1_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 2:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 5e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 1e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 1e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 0.8  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 0.8  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 1  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_2_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 3:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 5e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 1e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 1e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 0.8  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 0.8  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 1  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_3_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 4:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 6e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 1  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 1  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 0.03  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_4_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 5:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 6e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 1  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 1  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 0.045  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_5_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 6:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v5"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 6e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__["m_vf_coef"] = 1
            RUNHEADER.__dict__["m_vf_coef_2"] = 1
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__["m_pi_coef"] = 0.5
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_6_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 7:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v6"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 2e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 6e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = True  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 300
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__["m_vf_coef"] = 1
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 0.7  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m7_1_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 8:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v7"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 5e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 1e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 1e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__["m_vf_coef"] = 0.8
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 1  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_2_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 9:
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v8"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 1e-4
            RUNHEADER.__dict__["cyclic_lr_max"] = 5e-4

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["c_epoch"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 1e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 1e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__["m_vf_coef"] = 0.8
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 1  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m5_3_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        elif args.search_parameter == 99:  # fixed - 주식형 20 Projection - Bias version
            RUNHEADER.__dict__["default_net"] = "shake_regulization_v1"
            RUNHEADER.__dict__["m_n_cpu"] = 32  # a fixed n_cpu for nature_cnn_D
            RUNHEADER.__dict__["m_offline_learning_epoch"] = 900
            RUNHEADER.__dict__["warm_up_update"] = 1000
            RUNHEADER.__dict__["cosine_lr"] = True
            RUNHEADER.__dict__["cyclic_lr_min"] = 2e-4 * 3.5
            RUNHEADER.__dict__["cyclic_lr_max"] = 2e-4 * 5

            RUNHEADER.__dict__["m_on_validation"] = False
            RUNHEADER.__dict__["dynamic_lr"] = False  # made a decision True -> False
            RUNHEADER.__dict__["dynamic_coe"] = False  # made a decision True -> False
            RUNHEADER.__dict__["grad_norm"] = False
            RUNHEADER.__dict__["predefined_fixed_lr"] = [2e-4 * 3, 2e-4 * 3, 2e-4 * 3]
            RUNHEADER.__dict__["m_validation_interval"] = 600
            RUNHEADER.__dict__["m_validation_min_epoch"] = 0
            RUNHEADER.__dict__["m_learning_rate"] = 4e-4  # a made decision
            # 5e-5 -> 5e-4 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__[
                "m_offline_learning_rate"
            ] = 2e-4  # a made decision 5e-4 -> 3e-4 -> 5e-6 -> 5e-5 -> 5e-4
            RUNHEADER.__dict__["m_min_learning_rate"] = 2e-4  # 7e-6 ->  1e-5
            RUNHEADER.__dict__[
                "m_vf_coef"
            ] = 1  # 1 -> 0.2 -> [wow] 0.02 -> 0.01  # a made decision Note. Maybe reduce value again, loss converge too fast
            RUNHEADER.__dict__["m_vf_coef_2"] = 1  # 1 -> 0.2 -> [wow] 0.02
            RUNHEADER.__dict__["m_ent_coef"] = 0
            RUNHEADER.__dict__[
                "m_pi_coef"
            ] = 0.85  # 0.01 -> 1 -> 0.05 -> 0.1 -> 0.024 -> 0.005
            RUNHEADER.__dict__[
                "m_max_grad_norm"
            ] = 0.5  # [0.5 | None]  # a made decision
            RUNHEADER.__dict__[
                "m_l2_norm"
            ] = 1e-7  # 0.1 -> 4e-5 -> 4e-6 -> 1e-7(id3) -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_l1_norm"] = 1e-05
            RUNHEADER.__dict__[
                "m_drop_out"
            ] = 0.8  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_decay"
            ] = 0.9997  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__[
                "m_batch_epsilon"
            ] = 0.001  # -> inception_utils.inception_arg_scope()
            RUNHEADER.__dict__["m_name"] = (
                dict_RUNHEADER["m_name"]
                + "_m1_99_"
                + str(RUNHEADER.__dict__["dataset_version"])
                + "_"
                + time_now
                + "_"
                + str(args.process_id)
            )
        else:  # code test
            RUNHEADER.__dict__["default_net"] = "s3dg_v1"
        recent_procedure("./agent_log/working_model_p", args.process_id, "w")

    print(f"model name: {RUNHEADER.m_name}")
    return selected_x_dict, json_location


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


def init_start(header) -> None:
    for key in header.keys():
        RUNHEADER.__dict__[key] = header[key]
