from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import header.market_timing.RUNHEADER as RUNHEADER  # case: indexforecasting


def configure_header(args):
    def _print():
        print('\n===Load info===')
        print('X: {}'.format(RUNHEADER.raw_x))
        print('X_2: {}'.format(RUNHEADER.raw_x2))
        print('Y: {}'.format(RUNHEADER.target_name))
        print('dataset_version: {}'.format(RUNHEADER.dataset_version))
        print('gen_var: {}'.format(RUNHEADER.gen_var))
        print('use_c_name: {}'.format(RUNHEADER.use_c_name))
        print('use_var_mask: {}'.format(RUNHEADER.use_var_mask))
        print('objective: {}'.format(RUNHEADER.objective))
        print('max_x: {}'.format(RUNHEADER.max_x))
        print('s_test: {}'.format(RUNHEADER.s_test))
        print('e_test: {}'.format(RUNHEADER.e_test))

    def get_file_name(m_target_index, file_data_vars):
        return file_data_vars + RUNHEADER.target_id2name(m_target_index) + '_intermediate.csv'

    RUNHEADER.__dict__['dataset_version'] = args.dataset_version
    RUNHEADER.__dict__['m_target_index'] = args.m_target_index
    RUNHEADER.__dict__['target_name'] = RUNHEADER.target_id2name(args.m_target_index)
    RUNHEADER.__dict__['raw_y'] = './datasets/rawdata/index_data/gold_index.csv'
    RUNHEADER.__dict__['raw_x'] = None
    RUNHEADER.__dict__['raw_x2'] = None
    RUNHEADER.__dict__['use_c_name'] = None
    RUNHEADER.__dict__['use_var_mask'] = None
    RUNHEADER.__dict__['gen_var'] = None
    RUNHEADER.__dict__['max_x'] = None
    RUNHEADER.__dict__['s_test'] = args.s_test
    RUNHEADER.__dict__['e_test'] = args.e_test
    RUNHEADER.__dict__['m_target_index'] = args.m_target_index
    RUNHEADER.__dict__['var_desc'] = './datasets/rawdata/index_data/Synced_D_Summary.csv'

    if RUNHEADER.dataset_version == 'v0':
        RUNHEADER.__dict__['gen_var'] = args.gen_var
        if RUNHEADER.__dict__['gen_var']:
            RUNHEADER.__dict__['raw_x'] = './datasets/rawdata/index_data/Synced_D_FilledData_new_097.csv'  # th > 0.97 (memory error for US10YT)
            RUNHEADER.__dict__['raw_x'] = './datasets/rawdata/index_data/Synced_D_FilledData_new_' + str(RUNHEADER.derived_vars_th[0]) + '.csv'
            RUNHEADER.__dict__['raw_x2'] = './datasets/rawdata/index_data/Synced_D_FilledData.csv'  # whole data
        else:
            RUNHEADER.__dict__['raw_x'] = get_file_name(RUNHEADER.m_target_index,
                                                        './datasets/rawdata/index_data/data_vars_')
            RUNHEADER.__dict__['max_x'] = 500

    else:
        # online tf_record e.g. v10, v11, v12 ..., v21
        # RUNHEADER.__dict__['m_target_index'] = 1
        RUNHEADER.__dict__['use_c_name'] = True
        RUNHEADER.__dict__['use_var_mask'] = True
        RUNHEADER.__dict__['raw_x'] = './datasets/rawdata/index_data/Synced_D_FilledData.csv'
        RUNHEADER.__dict__['max_x'] = 150  # US10YR 변경 사항 반영 전, KS11, Gold, S&P 는 이 세팅으로 실험 결과 산출 함
        RUNHEADER.__dict__['max_x'] = 500  # 200으로 변경 함, 1. 변경 실험결과 산출 필요 2. 네트워크 파라미터 변경이 필요 할 수 도 있음.

    # re-assign
    RUNHEADER.__dict__['target_name'] = RUNHEADER.target_id2name(RUNHEADER.__dict__['m_target_index'])
    _print()
    return None, None
