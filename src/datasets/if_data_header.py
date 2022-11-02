# case: indexforecasting
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""
from __future__ import absolute_import, division, print_function

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

    def get_file_name(m_target_index, file_data_vars) -> str:
        return (
            file_data_vars
            + RUNHEADER.target_id2name(m_target_index)
            + "_intermediate.csv"
        )

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
            RUNHEADER.__dict__[
                "raw_x"
            ] = "./datasets/rawdata/index_data/Synced_D_FilledData_new_097.csv"  # th > 0.97 (memory error for US10YT)
            # # Disable - Generate derived vars
            # Data set for derived vars calculation
            # RUNHEADER.__dict__["raw_x"] = (
            #     "./datasets/rawdata/index_data/Synced_D_FilledData_new_"
            #     + str(RUNHEADER.derived_vars_th[0])
            #     + ".csv"
            # )
            RUNHEADER.__dict__[
                "raw_x2"
            ] = "./datasets/rawdata/index_data/Synced_D_FilledData.csv"  # whole data
        else:
            RUNHEADER.__dict__["raw_x"] = get_file_name(
                RUNHEADER.m_target_index, "./datasets/rawdata/index_data/data_vars_"
            )
            RUNHEADER.__dict__["max_x"] = 500

    else:
        # online tf_record e.g. v10, v11, v12 ..., v21
        # RUNHEADER.__dict__['m_target_index'] = 1
        RUNHEADER.__dict__["use_c_name"] = True
        RUNHEADER.__dict__["use_var_mask"] = True
        RUNHEADER.__dict__[
            "raw_x"
        ] = "./datasets/rawdata/index_data/Synced_D_FilledData.csv"
        RUNHEADER.__dict__[
            "max_x"
        ] = 150  # US10YT 변경 사항 반영 전, KS11, Gold, S&P 는 이 세팅으로 실험 결과 산출 함
        RUNHEADER.__dict__[
            "max_x"
        ] = 500  # 500으로 변경 함, 1. 변경 실험결과 산출 필요 2. 네트워크 파라미터 변경이 필요 할 수 도 있음.

    # re-assign
    RUNHEADER.__dict__["target_name"] = RUNHEADER.target_id2name(
        RUNHEADER.__dict__["m_target_index"]
    )
    _print()
