# case: indexforecasting
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""
from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Tuple

from header.index_forecasting import RUNHEADER


def configure_header(args) -> None:
    def _print() -> None:
        print("\n===Load info===")
        print(f"X: {RUNHEADER.raw_x}")
        print(f"X_2: {RUNHEADER.raw_x2}")
        print(f"Y: {RUNHEADER.target_name}")
        print(f"dataset_version: {RUNHEADER.dataset_version}")
        print(f"gen_var: {RUNHEADER.gen_var}")
        print(f"use_c_name: {RUNHEADER.use_c_name}")
        print(f"use_var_mask: {RUNHEADER.use_var_mask}")
        print(f"objective: {RUNHEADER.objective}")
        print(f"max_x: {RUNHEADER.max_x}")
        print(f"s_test: {RUNHEADER.s_test}")
        print(f"e_test: {RUNHEADER.e_test}")

    RUNHEADER.__dict__["dataset_version"] = args.dataset_version
    RUNHEADER.__dict__["m_target_index"] = args.m_target_index
    RUNHEADER.__dict__["target_name"] = RUNHEADER.target_id2name(args.m_target_index)
    RUNHEADER.__dict__["raw_y"] = "./datasets/rawdata/index_data/gold_index.csv"
    RUNHEADER.__dict__["raw_x"] = None
    RUNHEADER.__dict__["raw_x2"] = None
    RUNHEADER.__dict__["use_c_name"] = None
    RUNHEADER.__dict__["use_var_mask"] = None
    RUNHEADER.__dict__["gen_var"] = None
    RUNHEADER.__dict__["max_x"] = None
    RUNHEADER.__dict__["s_test"] = args.s_test
    RUNHEADER.__dict__["e_test"] = args.e_test
    RUNHEADER.__dict__["m_target_index"] = args.m_target_index
    RUNHEADER.__dict__[
        "var_desc"
    ] = "./datasets/rawdata/index_data/Synced_D_Summary.csv"

    if RUNHEADER.dataset_version == "v0":
        RUNHEADER.__dict__["gen_var"] = args.gen_var
        if RUNHEADER.__dict__["gen_var"]:
            assert False, "this features have been disabled !! (gen_var=1)"
        else:
            RUNHEADER.__dict__[
                "raw_x"
            ] = "./datasets/rawdata/index_data/Synced_D_FilledData.csv"
            RUNHEADER.__dict__["max_x"] = 500

    else:
        RUNHEADER.__dict__["use_c_name"] = True
        RUNHEADER.__dict__["use_var_mask"] = True
        RUNHEADER.__dict__[
            "raw_x"
        ] = "./datasets/rawdata/index_data/Synced_D_FilledData.csv"
        RUNHEADER.__dict__["max_x"] = 500

    # re-assign
    RUNHEADER.__dict__["target_name"] = RUNHEADER.target_id2name(
        RUNHEADER.__dict__["m_target_index"]
    )
    _print()
