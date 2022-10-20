from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import header.market_timing.RUNHEADER as RUNHEADER
import sc_parameters as scp

if RUNHEADER.release:
    from libs import market_timing_adhoc
    from libs.datasets import generate_val_test_with_X
else:
    import market_timing_adhoc
    from datasets import generate_val_test_with_X
from datasets.market_timing_protobuf2pickle import DataSet
import datasets.market_timing_protobuf2pickle as market_timing_protobuf2pickle

import market_timing_test
from multiprocessing.managers import BaseManager

import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")
import shutil
import argparse
import datetime
import util
from util import get_domain_on_CDSW_env
import numpy as np

def refine_jason_list(zip_info, MAX_HISTORICAL_MODELS=5):
    tmp_json_location_list = list()
    tmp_json_location_list_2 = list()
    for it in zip_info:
        if (it[3]==1) or (it[3]==True):
            tmp_json_location_list.append(it[:-1])  # tuple is hashable, but list and dict
        else:
            tmp_json_location_list_2.append(it[:-1])

    if len(tmp_json_location_list_2) > MAX_HISTORICAL_MODELS:
        tmp_json_location_list_2 = tmp_json_location_list_2[-MAX_HISTORICAL_MODELS:]
    aa = np.array(list(set(tmp_json_location_list + tmp_json_location_list_2)))
    return aa[:, 0].tolist(), aa[:, 1].tolist(), np.where(aa[:, 2]=='False', False, True).tolist()


def get_f_model_from_base(model_results, base_f_model):
    items = [item for item in os.listdir(model_results) if ".csv" in item]
    for item in items:
        if (
            base_f_model.split("_sub_epo_")[1].split("_")[0]
            == item.split("_sub_epo_")[1].split("_")[0]
        ):
            return item


def run(
    args, json_location, time_now, candidate_model, selected_model, performence_stacks
):
    dict_RUNHEADER = util.json2dict(
        "./save/model/rllearn/{}/agent_parameter.json".format(json_location)
    )

    # re-load
    current_release = RUNHEADER.release
    for key in dict_RUNHEADER.keys():
        RUNHEADER.__dict__[key] = dict_RUNHEADER[key]
    RUNHEADER.__dict__["m_final_model"] = f_test_model
    RUNHEADER.__dict__["m_bound_estimation"] = False
    RUNHEADER.__dict__["m_bound_estimation_y"] = True
    RUNHEADER.__dict__["release"] = current_release
    RUNHEADER.__dict__["dataset_version"] = args.dataset_version
    RUNHEADER.__dict__[
        "m_dataset_dir"
    ] = "./save/tf_record/market_timing/mt_x0_20_y{}_{}".format(
        args.forward_ndx, args.dataset_version
    )
    # additional dataset info
    RUNHEADER.__dict__[
        "raw_x"
    ] = "./datasets/rawdata/index_data/Synced_D_FilledData.csv"
    RUNHEADER.__dict__["raw_y"] = "./datasets/rawdata/index_data/gold_index.csv"
    RUNHEADER.__dict__[
        "var_desc"
    ] = "./datasets/rawdata/index_data/Synced_D_Summary.csv"
    RUNHEADER.__dict__["s_test"] = None
    RUNHEADER.__dict__["e_test"] = None
    RUNHEADER.__dict__["use_var_mask"] = True
    # RUNHEADER.__dict__["m_warm_up_4_inference"] = RUNHEADER.forward_ndx
    # RUNHEADER.__dict__["m_warm_up_4_inference"] = 6

    pickable_header = market_timing_test.convert_pickable(RUNHEADER)

    m_name = RUNHEADER.m_name
    m_inference_buffer = RUNHEADER.m_inference_buffer
    _model_location = "./save/model/rllearn/" + m_name
    _tensorboard_log = "./save/tensorlog/market_timing/" + m_name
    _dataset_dir = RUNHEADER.m_dataset_dir
    _full_tensorboard_log = False

    target_list = None
    _result_sub1 = "./save/result/{}".format(time_now)
    _result_sub2 = "./save/result/{}/{}_T{}".format(
        time_now, RUNHEADER.target_id2name(args.m_target_index), str(args.forward_ndx)
    )
    _result = "{}/{}".format(_result_sub2, _model_location.split("/")[-1])
    if candidate_model is not None:
        candidate_model = candidate_model + [_result]

    target_list = [
        _result_sub1,
        _result_sub2,
        _result,
        _result + "/fig_index",
        _result + "/fig_bound",
        _result + "/fig_scatter",
        _result + "/fig_index/index",
        _result + "/fig_index/return",
        _result + "/fig_index/analytics",
        _result + "/validation",
        _result + "/validation/fig_index",
        _result + "/validation/fig_index/index",
        _result + "/validation/fig_index/return",
        _result + "/final",
    ]

    for target in target_list:
        if not os.path.isdir(target):
            try:
                os.mkdir(target)
            except FileExistsError:
                print("try one more time")
                os.mkdir(target)
    copy_file = [
        "/selected_x_dict.json",
        "/agent_parameter.json",
        "/agent_parameter.txt",
        "/shuffled_episode_index.txt",
    ]

    # use predefined base model
    dir_name = _model_location
    [
        shutil.copy2(dir_name + file_name, target_list[-7] + file_name)
        for file_name in copy_file
    ]

    # check _dataset_dir in operation mode
    (
        _n_step,
        _cv_number,
        _n_cpu,
        _env_name,
        _file_pattern,
        _infer_set,
    ) = market_timing_test.meta_info(_model_location, _dataset_dir)

    _env_name = "MT-{}".format(RUNHEADER.dataset_version)
    _file_pattern = "mt_{}_cv%02d_%s.pkl".format(RUNHEADER.dataset_version)

    """run application"""
    assert (
        True if _model_location is not None else False
    ), "Require X list to generate val and test data set"
    x_dict, s_test, e_test, forward_ndx = market_timing_protobuf2pickle.get_x_dates(
        _model_location, _dataset_dir, RUNHEADER.forward_ndx
    )
    assert RUNHEADER.forward_ndx == forward_ndx, "Forward_ndx should be the same"

    print("\n[{}] Data Set Creation ...".format(m_name))
    tf_val, tf_test = generate_val_test_with_X.run(
        x_dict, s_test, e_test, None, "market_timing", forward_ndx
    )

    # register
    BaseManager.register("DataSet", DataSet)
    manager = BaseManager()
    manager.start(market_timing_test.init_start, (pickable_header,))
    exp_result = None
    for _mode in _infer_set:
        patch_data = tf_test if _mode == "test" else tf_val

        # dataset injection
        print("\n[{}] Data Set Load ...".format(_mode))
        sc = market_timing_test.Script(
            so=manager.DataSet(
                dataset_dir=_dataset_dir,
                file_pattern=_file_pattern,
                split_name=_mode,
                cv_number=_cv_number,
                regenerate=True,
                model_location=_model_location,
                forward_ndx=RUNHEADER.forward_ndx,
                patch_data=patch_data,
            )
        )

        print("[{}] Inference ...".format(_mode))
        if _mode == "validation":
            exp_result = "{}/validation".format(_result)
        else:
            exp_result = _result
        sc.run(
            mode=_mode,
            env_name=_env_name,
            tensorboard_log=_tensorboard_log,
            full_tensorboard_log=_full_tensorboard_log,
            model_location=_model_location,
            n_cpu=1,
            n_step=_n_step,
            result=exp_result,
            m_inference_buffer=m_inference_buffer,
            b_naive=False,
        )

    if candidate_model is not None:
        selected_model, performence_stacks = util.f_error_test(
            candidate_model, selected_model, performence_stacks
        )

    return (
        selected_model,
        {
            "_env_name": _env_name,
            "_cv_number": _cv_number,
            "_n_cpu": _n_cpu,
            "_n_step": _n_step,
            "exp_result": exp_result,
        },
        performence_stacks,
    )


if __name__ == "__main__":
    try:
        time_now = (
            str(datetime.datetime.now())[:-10]
            .replace(":", "-")
            .replace("-", "")
            .replace(" ", "_")
        )
        """configuration
        """
        parser = argparse.ArgumentParser("")
        # init args
        parser.add_argument("--process_id", type=int, default=1)
        parser.add_argument("--domain", type=str, required=True)
        parser.add_argument("--actual_inference", type=int, default=0)
        parser.add_argument("--m_target_index", type=int, default=None)
        parser.add_argument("--forward_ndx", type=int, default=None)
        parser.add_argument("--dataset_version", type=str, default=None)
        parser.add_argument("--performed_date", type=str, default=None)
        # # For Demo
        # parser.add_argument('--process_id', type=int, default=4)
        # parser.add_argument('--m_target_index', type=int, default=None)
        # parser.add_argument('--forward_ndx', type=int, default=None)
        # parser.add_argument('--actual_inference', type=int, default=0)
        # parser.add_argument('--dataset_version', type=str, default=None)
        # parser.add_argument("--domain", type=str, default='INX_20')
        # parser.add_argument("--performed_date", type=str, default=None)
        args = parser.parse_args()
        args.domain = get_domain_on_CDSW_env(args.domain)
        if args.actual_inference == 1:
            args.process_id = 1
        args = scp.ScriptParameters(
            args.domain, args, job_id_int=args.process_id
        ).update_args()

        if args.performed_date is not None:
            time_now = ''.join(performed_date.split('-'))
        enable_confidence = False  # Disalbe for the sevice, (computation cost issue)
        # re-write RUNHEADER
        if bool(args.actual_inference):
            (
                json_location_list,
                f_test_model_list,
                current_period_list,
                init_model_repo_list,
            ) = market_timing_test.get_model_from_meta_repo(
                RUNHEADER.target_id2name(args.m_target_index),
                str(args.forward_ndx),
                RUNHEADER.use_historical_model,
            )
            if type(json_location_list) is str:
                json_location_list, f_test_model_list, current_period_list, init_model_repo_list = (
                    [json_location_list],
                    [f_test_model_list],
                    [current_period_list],
                    [init_model_repo_list],
                )

            selected_model = None
            json_location_list, f_test_model_list, current_period_list = refine_jason_list(
                zip(json_location_list, f_test_model_list, current_period_list, init_model_repo_list), MAX_HISTORICAL_MODELS=5
            )
            performence_stacks = list()
            for idx in range(len(json_location_list) + 1):
                if idx < len(json_location_list):  # inference with candidate models
                    json_location, f_test_model, current_period = (
                        json_location_list[idx],
                        f_test_model_list[idx],
                        current_period_list[idx],
                    )
                    candidate_model = [json_location, f_test_model, current_period]

                    print("*****== Model Evaluation: {}".format(f_test_model))
                    selected_model, print_foot_note, performence_stacks = run(
                        args,
                        json_location,
                        time_now,
                        candidate_model,
                        selected_model,
                        performence_stacks,
                    )
                else:  # final evaluation to calculate confidence score
                    print("[{}] Model Inference".format(f_test_model))
                    selected_model = None
                    # 0 - json_location, 1 - f_test_model, 2 - current_period, 3 - model_performence, 4 - metrics_mae, 5 - metrics_ratio, 6 - metrics_accuray
                    performence_stacks = sorted(
                        performence_stacks, key=lambda item: item[4]
                    )
                    th_ratio, tf_accuracy = 0.32, 0.6
                    while (selected_model is None) and (th_ratio > 0.25):
                        for item in performence_stacks:
                            item[2] = False if item[2] == "False" else True

                            if item[2]:  # current best
                                selected_model = [item[0], item[1], item[2], item[3]]
                            else:
                                if item[5] >= th_ratio and item[6] >= tf_accuracy:
                                    selected_model = [
                                        item[0],
                                        item[1],
                                        item[2],
                                        item[3],
                                    ]
                        th_ratio = th_ratio - 0.01
                        tf_accuracy = tf_accuracy - 0.01
                    if selected_model is None:
                        # pick an alternative one - the worst case picking a latest model for the sevice
                        selected_model = performence_stacks[-1][:4]
                    (
                        json_location,
                        f_test_model,
                        current_period,
                        _result,
                    ) = selected_model

                    # final model evaluation with confidence score
                    if enable_confidence:
                        f_test_model = None  # Disable to calculate confidence score
                        _, print_foot_note, _ = run(
                            args, json_location, time_now, None, selected_model
                        )  # inference - get performences to calculate confidence score for the final model

                    # adhoc-process - confidence and align reg and classifier
                    target_name = RUNHEADER.target_id2name(args.m_target_index)
                    domain_detail = "{}_T{}_{}".format(
                        target_name, str(args.forward_ndx), args.dataset_version
                    )
                    domain = "{}_T{}".format(target_name, str(args.forward_ndx))
                    t_info = "{}_{}".format(domain, time_now)
                    target_file = "./save/model_repo_meta/{}.pkl".format(domain)
                    meta_list = market_timing_adhoc.load(target_file, "pickle")
                    meta = [
                        meta for meta in meta_list if meta["m_name"] == json_location
                    ][-1]

                    # result: 후처리 결과 파일 떨어지는 위치, model_location: 에폭별 실험결과 위치, f_base_model
                    print("\n*****== Adhoc Process: {}".format(_result))
                    sc = market_timing_adhoc.Script(
                        result=_result + "/final",
                        model_location=_result,
                        f_base_model=meta["m_name"],
                        f_model=get_f_model_from_base(_result, meta["model_name"]),
                        adhoc_file="AC_Adhoc.csv",
                        infer_mode=True,
                        info=t_info,
                    )
                    pd.set_option("mode.chained_assignment", None)
                    sc.run_adhoc()
                    pd.set_option("mode.chained_assignment", "warn")

                # print test environments
                if enable_confidence:
                    print("\nEnvs ID: {}".format(print_foot_note["_env_name"]))
                    print("Data Set Number: {}".format(print_foot_note["_cv_number"]))
                    print("Num Agents: {}".format(print_foot_note["_n_cpu"]))
                    print("Num Step: {}".format(print_foot_note["_n_step"]))
                    print("Result Directory: {}".format(print_foot_note["exp_result"]))
        else:
            """
            Intermediate model inference section to evaluate the model performence for a current best model
            before the final decision with a model repositories
            """
            market_timing_test.configure_header(args)
            pickable_header = market_timing_test.convert_pickable(RUNHEADER)

            m_name = RUNHEADER.m_name
            m_inference_buffer = RUNHEADER.m_inference_buffer
            _model_location = "./save/model/rllearn/" + m_name
            _tensorboard_log = "./save/tensorlog/market_timing/" + m_name
            _dataset_dir = RUNHEADER.m_dataset_dir
            _full_tensorboard_log = False

            target_list = None
            _result = "./save/result"
            _result = "{}/{}".format(_result, _model_location.split("/")[-1])
            target_list = [
                _result,
                _result + "/fig_index",
                _result + "/fig_bound",
                _result + "/fig_scatter",
                _result + "/fig_index/index",
                _result + "/fig_index/return",
                _result + "/fig_index/analytics",
                _result + "/validation",
                _result + "/validation/fig_index",
                _result + "/validation/fig_index/index",
                _result + "/validation/fig_index/return",
            ]

            for target in target_list:
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                try:
                    os.mkdir(target)
                except FileExistsError:
                    print("try one more time")
                    os.mkdir(target)
            copy_file = [
                "/selected_x_dict.json",
                "/agent_parameter.json",
                "/agent_parameter.txt",
                "/shuffled_episode_index.txt",
            ]

            dir_name = "./save/model/rllearn/{}".format(
                market_timing_test.recent_procedure(
                    "./agent_log/working_model_p", args.process_id, "r"
                )
            )
            [
                shutil.copy2(dir_name + file_name, target_list[-6] + file_name)
                for file_name in copy_file
            ]

            # check _dataset_dir in operation mode
            (
                _n_step,
                _cv_number,
                _n_cpu,
                _env_name,
                _file_pattern,
                _infer_set,
            ) = market_timing_test.meta_info(_model_location, _dataset_dir)

            exp_result = None
            for _mode in _infer_set:
                """run application"""
                # register
                BaseManager.register("DataSet", DataSet)
                manager = BaseManager()
                manager.start(market_timing_test.init_start, (pickable_header,))

                if _mode == "validation":
                    exp_result = "{}/validation".format(_result)
                else:
                    exp_result = _result

                # dataset injection
                sc = market_timing_test.Script(
                    so=manager.DataSet(
                        dataset_dir=_dataset_dir,
                        file_pattern=_file_pattern,
                        split_name=_mode,
                        cv_number=_cv_number,
                    )
                )

                sc.run(
                    mode=_mode,
                    env_name=_env_name,
                    tensorboard_log=_tensorboard_log,
                    full_tensorboard_log=_full_tensorboard_log,
                    model_location=_model_location,
                    n_cpu=1,
                    n_step=_n_step,
                    result=exp_result,
                    m_inference_buffer=m_inference_buffer,
                )
            # print test environments
            print("\nEnvs ID: {}".format(_env_name))
            print("Data Set Number: {}".format(_cv_number))
            print("Num Agents: {}".format(_n_cpu))
            print("Num Step: {}".format(_n_step))
            print("Result Directory: {}".format(exp_result))

    except Exception as e:
        print("\n{}".format(e))
        exit(1)
