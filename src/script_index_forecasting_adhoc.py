# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
from datetime import datetime

import sc_parameters as scp
from header.index_forecasting import RUNHEADER
from index_forecasting_adhoc import update_model_pool
from util import get_domain_on_CDSW_env


def is_final_exist(tDir):
    t1, t2 = False, False
    for loc in os.listdir(tDir):
        if "IF_" in loc:
            t1 = True
        if "csv" in loc:
            t2 = True
    flag = 1 if t1 and t2 else 0
    return flag


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser("")
        # init
        parser.add_argument("--m_target_index", type=int, default=None)
        parser.add_argument("--forward_ndx", type=int, default=None)
        parser.add_argument("--dataset_version", type=str, default=None)
        parser.add_argument("--domain", type=str, required=True)
        parser.add_argument(
            "--performed_date",
            type=str,
            default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # # # Debug - adhoc process
        # parser.add_argument("--m_target_index", type=int, default=None)
        # parser.add_argument("--forward_ndx", type=int, default=None)
        # parser.add_argument("--dataset_version", type=str, default=None)
        # parser.add_argument("--domain", type=str, default="TOTAL_20")
        # parser.add_argument(
        #     "--performed_date",
        #     type=str,
        #     default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # )

        args = parser.parse_args()

        args.domain = get_domain_on_CDSW_env(args.domain)
        args = scp.ScriptParameters(args.domain, args).update_args()

        performed_date = (
            args.performed_date[:-10]
            .replace(":", "-")
            .replace("-", "")
            .replace(" ", "_")
        )

        flag = is_final_exist(
            f"./save/result/selected/{'_T'.join(args.domain.split('_'))}/final"
        )

        update_model_pool(
            int(args.m_target_index),
            int(args.forward_ndx),
            str(args.dataset_version),
            flag,
            performed_date,
        )

    except Exception as e:
        print(f"\n{e}")
        exit(1)
