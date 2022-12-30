#!/bin/bash

# open with binary. make sure file format is unix e.g vim -b filename
# :%s/^M//g (^M=Ctrl+v+m)

:<<'END'
# FTSE 1M 예제 
m_t=4
f_ndx=20

# 다음 두 변수는 고정 - 11 부터 시작
main_pid=$(($m_t+11))
ds_v=$(($m_t+11))

###############################################
# Disable - 기록을 위해 남겨 둠
python script_index_forecasting_generate_data.py --dataset_version=v0 --m_target_index=$m_t --gen_var=1
###############################################

# Binary Data 생성 (일반적인 운영 절차의 경우와 기간이 주어지는 경우 - 첫 번째 스크립트 사용)
python script_index_forecasting_generate_data.py --domain=INX_120 --verbose=3 --operation_mode=1
python script_index_forecasting_generate_data.py --domain=INX_120 --s_test=2017-01-02 --e_test=2019-04-02 --verbose=3 --operation_mode=0

# 학습 데이터 생성
python script_index_forecasting_train.py --domain=INX_120 --m_online_buffer=1 --search_parameter=0 --process_id=1 --on_cloud=0 --n_cpu=1

# 학습 및 평가
python script_index_forecasting_train.py --domain=INX_120 --process_id=1 --on_cloud=1 --n_cpu=0
python script_index_forecasting_test.py --domain=INX_120 --process_id=1 --actual_inference=0

# 모형 선택 및 후 처리
python script_index_forecasting_select_model.py --domain=INX_120
python script_index_forecasting_adhoc.py  --domain=INX_120
python script_plot_gif.py
python script_index_forecasting_test.py --domain=INX_120 --actual_inference=1

# Auto Clean
python script_auto_clean_envs.py --m_target_index=$m_t --forward_ndx=$f_ndx
END

:<<END
python script_index_forecasting_generate_data.py --domain=INX_120 --verbose=3 --operation_mode=1
python script_index_forecasting_train.py --domain=INX_120 --m_online_buffer=1 --search_parameter=0 --process_id=1 --on_cloud=0 --n_cpu=1
python script_index_forecasting_train.py --domain=INX_120 --process_id=1
python script_index_forecasting_test.py --domain=INX_120 --process_id=1
python script_index_forecasting_select_model.py --domain=INX_120
python script_index_forecasting_adhoc.py  --domain=INX_120
python script_index_forecasting_test.py --domain=INX_120 --actual_inference=1
python script_plot_gif.py
exit 0
END





