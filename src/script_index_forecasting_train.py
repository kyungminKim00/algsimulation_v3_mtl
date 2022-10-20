from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import header.index_forecasting.RUNHEADER as RUNHEADER
import sc_parameters as scp
from datasets.index_forecasting_protobuf2pickle import DataSet
import util
from util import get_domain_on_CDSW_env

import index_forecasting_train
from multiprocessing.managers import BaseManager
import shutil
import os
import pickle
import argparse
import datetime


if __name__ == "__main__":
    try:
        """configuration"""
        # RUNHEADER = __import__('header.index_forecasting.RUNHEADER', fromlist=['RUNHEADER'])

        time_now = (
            str(datetime.datetime.now())[:-10]
            .replace(":", "-")
            .replace("-", "")
            .replace(" ", "_")
        )

        parser = argparse.ArgumentParser("")
        # init args
        parser.add_argument("--m_online_buffer", type=int, default=0)
        parser.add_argument("--search_variables", type=int, default=0)
        parser.add_argument("--search_parameter", type=int, default=None)
        parser.add_argument("--process_id", type=int, required=True)
        parser.add_argument(
            "--on_cloud", type=int, default=1
        )  # for debug test, load chunks of samples or all samples
        parser.add_argument("--dataset_version", type=str, default=None)
        parser.add_argument("--n_cpu", type=int, default=0)
        parser.add_argument("--m_target_index", type=int, default=None)  # [0 | 1 | 2]
        parser.add_argument("--forward_ndx", type=int, default=None)  # [30 | 60 | 120]
        parser.add_argument("--ref_pid", type=int, default=None)
        parser.add_argument("--domain", type=str, required=True)

        # # Demo
        # parser.add_argument("--m_online_buffer", type=int, default=0)
        # parser.add_argument("--search_variables", type=int, default=0)
        # parser.add_argument("--search_parameter", type=int, default=None)
        # parser.add_argument("--process_id", type=int, default=4)
        # parser.add_argument(
        #     "--on_cloud", type=int, default=1
        # )  # for debug test, load chunks of samples or all samples
        # parser.add_argument("--dataset_version", type=str, default=None)
        # parser.add_argument("--n_cpu", type=int, default=0)
        # parser.add_argument("--m_target_index", type=int, default=None)  # [0 | 1 | 2]
        # parser.add_argument("--forward_ndx", type=int, default=None)
        # parser.add_argument("--ref_pid", type=int, default=None)
        # parser.add_argument("--domain", type=str, default='INX_20')
        args = parser.parse_args()
        args.domain = get_domain_on_CDSW_env(args.domain)
        args = scp.ScriptParameters(args.domain, args, job_id_int=args.process_id, search_parameter=args.search_parameter).update_args()

        if args.m_online_buffer == 1:
            args.process_id = args.ref_pid
            args.ref_pid = 0

        if bool(args.ref_pid):
            assert args.m_online_buffer == 0, "{}: check your parameters".format(
                __name__
            )
            if args.ref_pid == args.process_id:
                pass
            else:
                shutil.copy2(
                    "./agent_log/buffer_generate_model_p{}.txt".format(args.ref_pid),
                    "./agent_log/buffer_generate_model_p{}.txt".format(args.process_id),
                )

        # re-write RUNHEADER
        (
            RUNHEADER.__dict__["m_target_index"],
            RUNHEADER.__dict__["target_name"],
            RUNHEADER.__dict__["m_name"],
        ) = RUNHEADER.init_var(args)
        selected_x_dict, json_location = index_forecasting_train.configure_header(args)
        pickable_header = index_forecasting_train.convert_pickable(
            RUNHEADER
        )  # win32 only support spawn method for multiprocess unlike linux

        print("m_online_buffer: {}".format(RUNHEADER.m_online_buffer))
        print("search_variables: {}".format(RUNHEADER.search_variables))
        print("search_parameter: {}".format(RUNHEADER.search_parameter))
        print("process_id: {}".format(str(args.process_id)))
        print("dataset_version: {}".format(RUNHEADER.dataset_version))
        print("m_offline_buffer_file: {}".format(RUNHEADER.m_offline_buffer_file))
        print("m_name: {}".format(RUNHEADER.m_name))
        print("forecast: {}".format(RUNHEADER.forward_ndx))
        print("target: {}".format(RUNHEADER.target_name))

        m_name = RUNHEADER.m_name
        _n_step = RUNHEADER.m_n_step
        _n_cpu = RUNHEADER.m_n_cpu
        _total_timesteps = RUNHEADER.m_total_timesteps
        _learning_rate = RUNHEADER.m_learning_rate
        _verbose = RUNHEADER.m_verbose
        _log_interval = RUNHEADER.m_tabular_log_interval

        _model_location = "./save/model/rllearn/" + m_name
        _tensorboard_log = (
            "./save/tensorlog/" + RUNHEADER.tf_record_location + "/" + m_name
        )

        # mkdir for model, log, and result
        target = None
        for k in range(3):
            if k == 0:
                target = _model_location
            elif k == 1:
                target = _tensorboard_log
            elif k == 2:
                location = _model_location.split("/")
                target = "{}/{}/result/{}".format(location[0], location[1], location[4])

            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            os.mkdir(target)

        # mkdir for validation
        validation = _model_location + "/validation"
        target_list = [
            validation,
            validation + "/fig_index",
            validation + "/fig_bound",
            validation + "/fig_scatter",
            validation + "/fig_index/index",
            validation + "/fig_index/return",
            validation + "/fig_index/analytics",
        ]
        for target in target_list:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            try:
                os.mkdir(target)
            except FileExistsError:
                print("try one more time")
                os.mkdir(target)

        # mkdir for buffer files
        if RUNHEADER.m_online_buffer:
            if not os.path.isdir(RUNHEADER.m_offline_buffer_file):
                os.mkdir(RUNHEADER.m_offline_buffer_file)

        # copy configurations after creating folder
        if RUNHEADER.m_online_buffer:  # Generate Buffer
            if args.search_variables:  # random pick x variables
                util.dict2json(
                    _model_location + "/selected_x_dict.json", selected_x_dict
                )
            else:
                shutil.copy2(
                    RUNHEADER.m_dataset_dir + "/x_index.json",
                    _model_location + "/selected_x_dict.json",
                )
        else:  # Learning with Buffer
            copy_file = ["/selected_x_dict.json"]
            [
                shutil.copy2(
                    "./save/model/rllearn/" + json_location + file_name,
                    _model_location + file_name,
                )
                for file_name in copy_file
            ]
            # shutil.copy2('./save/model/rllearn/' + json_location + '/selected_x_dict.json',
            #              _model_location + '/selected_x_dict.json')
            # shutil.copy2('./save/model/rllearn/' + json_location + '/shuffled_episode_index.txt',
            #              _model_location + '/shuffled_episode_index.txt')
        ###

        _mode = "train"  # [train | validation | test]
        _full_tensorboard_log = RUNHEADER._full_tensorboard_log
        _dataset_dir = RUNHEADER.m_dataset_dir

        _env_name = str(RUNHEADER.objective) + "-" + str(RUNHEADER.dataset_version)
        # _env_name = RUNHEADER.m_env
        _file_pattern = (
            str(RUNHEADER.l_objective)
            + "_"
            + str(RUNHEADER.dataset_version)
            + "_cv%02d_%s.pkl"
        )  # data set file name
        # _file_pattern = RUNHEADER.m_file_pattern
        _cv_number = RUNHEADER.m_cv_number

        meta = {
            "_env_name": _env_name,
            "_model_location": _model_location,
            "_file_pattern": _file_pattern,
            "_n_step": _n_step,
            "_cv_number": _cv_number,
            "_n_cpu": _n_cpu,
        }

        # save number of environments
        with open(_model_location + "/meta", mode="wb") as fp:
            pickle.dump(meta, fp)
            fp.close()

        """ run application
        """
        BaseManager.register("DataSet", DataSet)
        if RUNHEADER.m_on_validation is True:
            BaseManager.register("DataSet_Validation", DataSet)
        manager = BaseManager()
        manager.start(index_forecasting_train.init_start, (pickable_header,))

        # dataset injection
        if RUNHEADER.m_on_validation is True:
            sc = index_forecasting_train.Script(
                so=manager.DataSet(
                    dataset_dir=_dataset_dir,
                    file_pattern=_file_pattern,
                    split_name=_mode,
                    cv_number=_cv_number,
                    n_batch_size=_n_step,
                ),
                so_validation=manager.DataSet_Validation(
                    dataset_dir=_dataset_dir,
                    file_pattern=_file_pattern,
                    split_name="validation",
                    cv_number=_cv_number,
                ),
            )
        else:
            sc = index_forecasting_train.Script(
                so=manager.DataSet(
                    dataset_dir=_dataset_dir,
                    file_pattern=_file_pattern,
                    split_name=_mode,
                    cv_number=_cv_number,
                    n_batch_size=_n_step,
                )
            )

        sc.run(
            mode=_mode,
            env_name=_env_name,
            tensorboard_log=_tensorboard_log,
            full_tensorboard_log=_full_tensorboard_log,
            model_location=_model_location,
            verbose=_verbose,
            n_cpu=_n_cpu,
            n_step=_n_step,
            learning_rate=_learning_rate,
            total_timesteps=_total_timesteps,
            log_interval=_log_interval,
        )
    except Exception as e:
        print("\n{}".format(e))
        exit(1)