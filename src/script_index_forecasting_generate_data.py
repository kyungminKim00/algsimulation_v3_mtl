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

if RUNHEADER.release:
    from libs.datasets import convert_if_v0  # Index_forecasting
    from libs.datasets import convert_if_v1  # Index_forecasting - Add mask (opt)
else:
    from datasets import convert_if_v0  # Index_forecasting
    from datasets import convert_if_v1  # Index_forecasting - Add mask (opt)

import tensorflow as tf
import argparse

from datasets.if_data_header import configure_header
from util import get_domain_on_CDSW_env


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError("You must supply the dataset name with --dataset_name")

    # if FLAGS.dataset_name == 'fs_x0_20_y5_v0':
    #     convert_fs_v0.run('./save/tf_record/fund_selection/fs_x0_20_y5_v0', 'fs_v0_cv%02d_%s.tfrecord')
    # elif FLAGS.dataset_name == 'fs_x0_20_y5_v1':
    #     convert_fs_v1.run('./save/tf_record/fund_selection/fs_x0_20_y5_v1', 'fs_v1_cv%02d_%s.tfrecord')
    # elif FLAGS.dataset_name == 'fs_x0_20_y5_v2':  # Complete
    #     convert_fs_v2.run('./save/tf_record/fund_selection/fs_x0_20_y5_v2', 'fs_v2_cv%02d_%s.tfrecord')
    # elif FLAGS.dataset_name == 'fs_x0_20_y5_v3':  # Complete
    #     convert_fs_v3.run('./save/tf_record/fund_selection/fs_x0_20_y5_v3', 'fs_v3_cv%02d_%s.tfrecord')
    # elif FLAGS.dataset_name == 'fs_x0_20_y5_v4':
    #     convert_fs_v4.run('./save/tf_record/fund_selection/fs_x0_20_y5_v4', 'fs_v4_cv%02d_%s.tfrecord')
    if "_v0" in FLAGS.dataset_name:
        convert_if_v0.run(
            "./save/tf_record/index_forecasting/" + FLAGS.dataset_name,
            "if_v0_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v0":
        convert_if_v0.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v0",
            "if_v0_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v1":
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v1",
            "if_v1_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v2":
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v2",
            "if_v2_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v3":
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v3",
            "if_v3_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v4":  # Merged_New stride 3, mask off
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v4",
            "if_v4_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v5":  # Merged_New stride 2, mask off
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v5",
            "if_v5_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif FLAGS.dataset_name == "if_x0_20_y20_v6":  # Merged_New stride 2, mask on
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v6",
            "if_v6_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif (
        FLAGS.dataset_name == "if_x0_20_y20_v7"
    ):  # Synced_D_FilledData stride 2, mask on, Gold
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v7",
            "if_v7_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif (
        FLAGS.dataset_name == "if_x0_20_y20_v8"
    ):  # Synced_D_FilledData stride 2, mask on, S&P
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v8",
            "if_v8_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    elif (
        FLAGS.dataset_name == "if_x0_20_y20_v9"
    ):  # Synced_D_FilledData stride 2, mask on, KOSPI
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/if_x0_20_y20_v9",
            "if_v9_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
        )
    else:
        # for online test
        convert_if_v1.run(
            "./save/tf_record/index_forecasting/" + FLAGS.dataset_name,
            "if_" + RUNHEADER.dataset_version + "_cv%02d_%s.tfrecord",
            s_test=FLAGS.s_test,
            e_test=FLAGS.e_test,
            verbose=FLAGS.verbose,
            _forward_ndx=int(FLAGS.forward_ndx),
            operation_mode=int(FLAGS.operation_mode),
            _performed_date=FLAGS.performed_date,
        )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser("")
        # init args
        parser.add_argument("--s_test", type=str, default=None)
        parser.add_argument("--e_test", type=str, default=None)
        parser.add_argument(
            "--dataset_version", type=str, default=None
        )  # save as v7 'v7'
        parser.add_argument("--verbose", type=int, default=None)
        parser.add_argument("--m_target_index", type=int, default=None)  # [0 | 1 | 2]
        parser.add_argument("--gen_var", type=int, default=None)  # [True | False]
        parser.add_argument("--forward_ndx", type=int, default=None)
        parser.add_argument("--operation_mode", type=int, default=None)
        parser.add_argument("--domain", type=str, required=True)
        parser.add_argument("--performed_date", type=str, default=None)

        # # Demo v0
        # parser.add_argument("--s_test", type=str, default=None)
        # parser.add_argument("--e_test", type=str, default=None)
        # parser.add_argument("--dataset_version", type=str, default='v0')
        # # [0: train/validation independent | 1: test | 2: train only | 3: train/validation Duplicate]
        # parser.add_argument("--verbose", type=int, default=None)
        # parser.add_argument("--m_target_index", type=int, default=4)  # [0 | 1 | 2]
        # parser.add_argument("--gen_var", type=int, default=1)  # [True | False]
        # parser.add_argument("--forward_ndx", type=int, default=None)
        # parser.add_argument("--operation_mode", type=int, default=None)
        # parser.add_argument("--domain", type=str, default=None)
        # parser.add_argument("--performed_date", type=str, default=None)

        # # for online test - Demo
        # parser.add_argument("--s_test", type=str, default=None)
        # parser.add_argument("--e_test", type=str, default=None)
        # parser.add_argument("--dataset_version", type=str, default=None)
        # # [0: train/validation independent | 1: test | 2: train only | 3: train/validation Duplicate]
        # parser.add_argument("--verbose", type=int, default=3)
        # parser.add_argument("--m_target_index", type=int, default=None)  # [0 | 1 | 2]
        # parser.add_argument("--gen_var", type=int, default=None)  # [True | False]
        # parser.add_argument("--forward_ndx", type=int, default=None)
        # parser.add_argument("--operation_mode", type=int, default=1)
        # parser.add_argument("--domain", type=str, default='INX_20')
        # parser.add_argument("--performed_date", type=str, default='2021-07-09')

        # # for online test - Demo
        # parser.add_argument("--s_test", type=str, default='2017-01-01')
        # parser.add_argument("--e_test", type=str, default='2017-04-01')
        # parser.add_argument("--dataset_version", type=str, default=None)
        # # [0: train/validation independent | 1: test | 2: train only | 3: train/validation Duplicate]
        # parser.add_argument("--verbose", type=int, default=3)
        # parser.add_argument("--m_target_index", type=int, default=None)  # [0 | 1 | 2]
        # parser.add_argument("--gen_var", type=int, default=None)  # [True | False]
        # parser.add_argument("--forward_ndx", type=int, default=None)
        # parser.add_argument("--operation_mode", type=int, default=0)
        # parser.add_argument("--domain", type=str, default='INX_20')
        # parser.add_argument("--performed_date", type=str, default=None)

        args = parser.parse_args()
        if args.dataset_version == "v0":
            assert (args.m_target_index is not None) and (
                args.gen_var is not None
            ), "the values of variables, m_target_index and gen_var, are required"
        else:
            args.domain = get_domain_on_CDSW_env(args.domain)
            args = scp.ScriptParameters(args.domain, args).update_args()

        (
            RUNHEADER.__dict__["m_target_index"],
            RUNHEADER.__dict__["target_name"],
            RUNHEADER.__dict__["m_name"],
        ) = RUNHEADER.init_var(args)
        _, _ = configure_header(args)

        FLAGS = tf.compat.v1.app.flags.FLAGS
        tf.compat.v1.app.flags.DEFINE_string(
            "s_test", args.s_test, "the start date of test data"
        )
        tf.compat.v1.app.flags.DEFINE_string(
            "e_test", args.e_test, "the end date of test data"
        )
        tf.compat.v1.app.flags.DEFINE_string(
            "verbose", str(args.verbose), "the end date of test data"
        )
        tf.compat.v1.app.flags.DEFINE_string(
            "forward_ndx", str(args.forward_ndx), "forward_ndx"
        )
        tf.compat.v1.app.flags.DEFINE_string(
            "operation_mode", str(args.operation_mode), "operation_mode"
        )
        tf.compat.v1.app.flags.DEFINE_string(
            "performed_date", str(args.performed_date), "performed_date"
        )

        dataset_version = None
        if RUNHEADER.objective == "FS":
            dataset_version = "fs_x0_20_y{}_{}".format(
                args.forward_ndx, RUNHEADER.dataset_version
            )
        if RUNHEADER.objective == "IF":
            dataset_version = "if_x0_20_y{}_{}".format(
                args.forward_ndx, RUNHEADER.dataset_version
            )
        if RUNHEADER.objective == "MT":
            dataset_version = "mt_x0_20_y{}_{}".format(
                args.forward_ndx, RUNHEADER.dataset_version
            )
        tf.compat.v1.app.flags.DEFINE_string(
            "dataset_name", dataset_version, "Data set name"
        )

        """
        # Merged_New stride 2
        # Multi-linearity & correlation y to x & masking
        1. 5일 이평, window 20일 간의 데이터를 분석한다. t 20일 분석한다(20 samples).
        2. 데이터 셑에 대해 상관계수를 산출한다. y to x
        3. 상관계수가 0에 가까운 x 변수는 0으로 마스킹 한다. (평균:0, 분산:0)
        4. 독립 변수간의 유클리디안 거리(t 상관계수 벡터 분석)를 활용하여 대표 변수를 찾고 그룹내 나머지 변수는 0으로 마스킹 한다.
        5. 마스킹이 0이 아닌 원소에 대해, |상관계수| 값을 마스크에 채워 넣는다.

        OPT. 
        [다중공선성] 1. 배치내의 W와 랜덤 벡터 R 간의 거리를 측정하여, W를 군집화하고 군집에 속한 갯수만큼 나누어 주어 Update가 덜 되도록한다(순전파에 영향을 덜 주도록 한다)
        [변수 선택 ] 2. X(입력)M(마스크 벡터)W(변수선택 가중치) + X(입력)W(컨볼루션 가중치) - 마스크의 가중치가 입력값을 왜곡 시키기 때문에 두 번째 텀을 더 하여 줌. 더 고민해 보기

            - 마스크의 가중치가 입력값을 왜곡 시키기 때문에 두 번째 텀을 더 하여 줌. 더 고민해 보기
            # (1[X] * 0.5[M]) 와 (0.5 * 1) 는 해석적으로 비대칭적이다. 
        """

        tf.compat.v1.app.run()
    except Exception as e:
        print("\n{}".format(e))
        exit(1)
