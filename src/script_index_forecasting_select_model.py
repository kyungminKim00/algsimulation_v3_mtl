from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import header.index_forecasting.RUNHEADER as RUNHEADER

if RUNHEADER.release:
    from libs import index_forecasting_select_model
else:
    import index_forecasting_select_model

import numpy as np

import os
import shutil
import argparse

# import pandas as pd
# from scipy.stats import entropy
# import pickle
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('agg')
#
# import re
# import sys
from util import check_training_status, get_domain_on_CDSW_env
import sc_parameters as scp


class TrainMoreError(Exception):
    def __init__(self, _str):
        print(_str)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser("")
        # init args
        parser.add_argument("--m_target_index", type=int, default=None)
        parser.add_argument("--forward_ndx", type=int, default=None)
        parser.add_argument("--dataset_version", type=str, default=None)
        parser.add_argument("--domain", type=str, required=True)
        # # Demo
        # parser.add_argument('--m_target_index', type=int, default=None)  # for operation mode
        # parser.add_argument('--forward_ndx', type=int, default=None)  # for operation mode
        # parser.add_argument('--dataset_version', type=str, default=None)
        # parser.add_argument("--domain", type=str, default='INX_20')
        args = parser.parse_args()
        args.domain = get_domain_on_CDSW_env(args.domain)
        args = scp.ScriptParameters(args.domain, args).update_args()

        m_target_index = args.m_target_index
        forward_ndx = str(args.forward_ndx)
        dataset_version = args.dataset_version
        target_name = RUNHEADER.target_id2name(m_target_index)

        # check training status
        print("== Running a model selection script")
        b_c_t_s, cnt = check_training_status(
            RUNHEADER.b_activate, RUNHEADER.r_model_cnt, target_name, forward_ndx
        )
        if not b_c_t_s:
            raise TrainMoreError(
                "The number of trained models are {}. at least {} trained models are required!! Keep Traning ... ".format(
                    str(cnt), str(RUNHEADER.r_model_cnt)
                )
            )

        roof = True
        index_result = (
            False  # remove to reduce drive spaces, enable when a operation mode
        )
        while roof:
            roof = False
            """configuration
            """
            max_cnt = 10  # 10 fixed
            soft_cond = True  # disable when operation mode. A decision making after a experimental
            select_criteria = 0.1  # Recommend value is at least positive but for the fast evaluation of models, the value is set to -2
            th_dict = {
                "th_pl": 1.85,
                "th_vl": 1.9,
                "th_ev": 0.95,
                "th_v_c": 0.9,
                "th_train_c_acc": 0.85,
                "th_v_mae": 8,
                "th_v_r_acc": 0.6,
                "th_v_ev": 0.5,
                "th_epoch": 200,
                "th_sub_score": -1,
            }

            # keep version
            th_dict = {
                "th_pl": np.random.permutation(np.arange(1.5, 2, 0.01))[0],
                "th_vl": np.random.permutation(np.arange(1.3, 1.9, 0.1))[0],
                "th_ev": np.random.permutation(np.arange(0.90, 1, 0.01))[0],
                "th_v_c": np.random.permutation(np.arange(0.90, 0.93, 0.01))[0],
                "th_train_c_acc": np.random.permutation(np.arange(0.85, 0.90, 0.01))[0],
                "th_v_mae": np.random.permutation(np.arange(3, 8, 1))[0],
                "th_v_r_acc": 0.6,
                "th_v_ev": 0.5,
                "th_epoch": 200,
                "th_sub_score": -1,
            }

            # new version
            th_dict = {
                "th_pl": 1.77,
                "th_vl": 1.8,
                "th_ev": 0.96,
                "th_v_c": 0.9,
                "th_train_c_acc": 0.85,
                "th_v_mae": 7,
                "th_v_r_acc": np.random.permutation(np.arange(0.5, 1, 0.1))[0],
                "th_v_ev": np.random.permutation(np.arange(0, 1, 0.1))[0],
                "th_epoch": 200,
                "th_sub_score": -1,
            }

            # ks version
            th_dict = {
                "th_pl": 1.77,
                "th_vl": 1.8,
                "th_ev": 0.90,
                "th_v_c": 0.9,
                "th_train_c_acc": 0.85,
                "th_v_mae": 7,
                "th_v_r_acc": 0.65,
                "th_v_ev": 0.6,
                "th_epoch": 150,  # default: 200, but KS11 has been trained with 200 epoch -> 150
                "th_sub_score": 0,
            }

            final_performance = list()
            ver_list = [
                [
                    it
                    for it in os.listdir("./save/result")
                    if target_name in it
                    and "T" + forward_ndx in it
                    and dataset_version in it
                ]
            ]

            if RUNHEADER.b_select_model_batch:
                ver_list = [
                    "v30",
                    "v31",
                    "v32",
                    "v33",
                    "v34",
                    "v35",
                    "v36",
                    "v37",
                    "v38",
                    "v39",
                    "v40",
                    "v41",
                    "v10",
                    "v11",
                    "v12",
                    "v13",
                    "v14",
                    "v15",
                    "v16",
                    "v17",
                    "v18",
                    "v19",
                    "v20",
                    "v21",
                    "v50",
                    "v51",
                    "v52",
                    "v53",
                    "v54",
                    "v55",
                    "v56",
                    "v57",
                    "v58",
                    "v59",
                    "v60",
                    "v61",
                ]

            flag = None
            init = True
            for ver in ver_list:
                # parser = argparse.ArgumentParser('')
                # parser.add_argument('--dataset_version', type=str, default=ver)
                # args = parser.parse_args()

                # rDir = './save/result/' + args.dataset_version
                # tDir = './save/result/selected/' + args.dataset_version

                rDir = ver
                bDir = "./save/result/selected/"
                tDir = "./save/result/selected/" + target_name + "_T" + forward_ndx
                if RUNHEADER.b_select_model_batch:
                    tDir = (
                        "./save/result/selected/" + ver + "_T" + forward_ndx
                    )  # batch test

                target_list = [bDir, tDir, tDir + "/final"]
                for target in target_list:
                    if os.path.isdir(target) and init:
                        if target != bDir:
                            shutil.rmtree(target, ignore_errors=True)
                            init = False
                    try:
                        if not os.path.isdir(target):
                            os.mkdir(target)
                    except FileExistsError:
                        print("try one more time")
                        os.mkdir(target)

                """ run application
                """
                sc = index_forecasting_select_model.Script(
                    tDir, rDir, max_cnt, select_criteria, soft_cond, th_dict
                )
                flag = sc.run_s_model(
                    dataset_version,
                    index_result=index_result,
                    b_batch_test=RUNHEADER.b_select_model_batch,
                )

                if flag == 0:
                    pass
                    # break

                # print test environments
                print("\ndataset_version: {}".format(dataset_version))
                print("source loc: {}".format(ver))
                print("target loc: {}".format(tDir))

                # stack final result
                base_model = ""
                for final_result in os.listdir(tDir + "/final"):
                    if (
                        "jpeg" not in final_result
                        and "R_" not in final_result
                        and "csv" not in final_result
                    ):
                        base_model = final_result
                        print("Base model: {}".format(base_model))
                    if "jpeg" in final_result and "R_" in final_result:
                        final_performance.append(final_result)
                        print(
                            "name: {} \n{}: {}\n".format(
                                final_result[:-5],
                                dataset_version,
                                float(final_result.split("_")[14]),
                            )
                        )

            # if flag == 1:
            #     index_forecasting_select_model.print_summary(final_performance, th_dict)
            index_forecasting_select_model.print_summary(final_performance, th_dict)
    except TrainMoreError:
        exit(2)
    except Exception as e:
        print("\n{}".format(e))
        exit(1)
