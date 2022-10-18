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
else:
    import market_timing_adhoc
import argparse
from util import get_domain_on_CDSW_env

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser("")
        # init
        parser.add_argument("--m_target_index", type=int, default=None)
        parser.add_argument("--forward_ndx", type=int, default=None)
        parser.add_argument("--dataset_version", type=str, default=None)
        parser.add_argument("--operation_mode", type=int, default=1)
        parser.add_argument("--domain", type=str, required=True)
        parser.add_argument("--init_repo_model", type=int, default=0)
        parser.add_argument("--performed_date", type=str, default=None)
        # # Demo
        # parser.add_argument('--m_target_index', type=int, default=0)
        # parser.add_argument('--forward_ndx', type=int, default=20)
        # parser.add_argument('--dataset_version', type=str, default='v11')
        # parser.add_argument('--operation_mode', type=int, default=1)
        # parser.add_argument("--domain", type=str, default=None)
        # parser.add_argument("--init_repo_model", type=int, default=0)
        # parser.add_argument("--performed_date", type=str, default=None)
        args = parser.parse_args()
        args.domain = get_domain_on_CDSW_env(args.domain)
        args = scp.ScriptParameters(args.domain, args).update_args()

        target_index = args.m_target_index
        forward_ndx = str(args.forward_ndx)
        dataset_version = args.dataset_version
        target_name = RUNHEADER.target_id2name(target_index)

        if args.operation_mode == 0:
            target_result = [
                [int(target_index), forward_ndx, k]
                for k in [
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
                    "v22",
                ]
            ]
        else:
            target_result = [[target_index, forward_ndx, dataset_version]]

        # adjusts return values for the consistency among heads and calculates confidence scores
        flag = list()
        for it in target_result:
            rn = market_timing_adhoc.Adhoc(it[0], it[1], it[2], args.performed_date)
            flag.append(rn.run())

        # 모델에서 파이널로 선택 되지 않는 것만 계속 지우고(지금은 학습후 다 지우는데 그거 수정), 파이널이 있느면 재 추론 없으면 모델풀에서 하나 선택해서 파이널 모형으로 강제 선택하고 재 추론
        if args.operation_mode:
            assert len(flag) == 1, "a len(flag) should be 1 on th operation mode"
            market_timing_adhoc.update_model_pool(
                target_index, forward_ndx, dataset_version, flag[0], args.init_repo_model
            )

        # calculate statics of performance of confidence scores - Experimental mode only
        if args.operation_mode == 0:
            operation_mode_simulation = True
            if operation_mode_simulation:
                for idx in range(len(target_result)):
                    market_timing_adhoc.update_model_pool(
                        target_result[idx][0],
                        target_result[idx][1],
                        target_result[idx][2],
                        flag[idx],
                        args.init_repo_model,
                    )
            else:
                print("Summay confidence performances")
                market_timing_adhoc.print_confidence_performance(
                    target_name, forward_ndx
                )
                print("Done: confidence_calibration")
    except Exception as e:
        print("\n{}".format(e))
        exit(1)
