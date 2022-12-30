#!/bin/bash

# open with binary. make sure file format is unix e.g vim -b filename
# :%s/^M//g (^M=Ctrl+v+m)

# S&P500
m_t=0
m_t=$1

# 1m 예측
f_ndx=20
f_ndx=$2

# 다음 두 변수는 시장 인덱스의 파생 변수로 고정
main_pid=$(($m_t+11))
ds_v=$(($m_t+11))

# 일반 일 배치 - 데이터 수집
/usr/bin/python3.6 script_generate_data.py --dataset_version=v$ds_v --verbose=3 --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=1

:<<'END'
# 기간 선정 데이터 수집
/usr/bin/python3.6 script_generate_data.py --dataset_version=v$ds_v --s_test=2018-01-02 --e_test=2018-04-02 --verbose=3 --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=0
END

:<<'END'
# 기간 선정 데이터 수집
/usr/bin/python3.6 script_generate_data.py --dataset_version=v$ds_v --s_test=$3 --e_test=$4 --verbose=3 --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=0
END

# 학습 데이터 샘플링
/usr/bin/python3.6 script_index_forecasting_train.py --m_online_buffer=1 --search_parameter=0 --process_id=$main_pid --on_cloud=0 --n_cpu=3 --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx

# 모형 학습 7번 병렬 수행
for i in 0 1 2 3 4 5 6
do
	/usr/bin/python3.6 script_index_forecasting_train.py --m_online_buffer=0 --search_parameter=1 --process_id=$(($main_pid+$i)) --on_cloud=0 --n_cpu=0 --m_target_index=$m_t --forward_ndx=$f_ndx --ref_pid=$main_pid
done

# 모형 평가 7번 병렬 수행
for k in 0 1 2 3 4 5 6
do
	/usr/bin/python3.6 script_index_forecasting_test.py --process_id=$(($main_pid+$k)) --operation_mode=0
done

# 모형 선택
/usr/bin/python3.6 script_index_forecasting_select_model.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx

# 후 처리
/usr/bin/python3.6 script_index_forecasting_adhoc.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=1

# 최종 예측 값 생성
/usr/bin/python3.6 script_index_forecasting_test.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=1

# 임시 생성 데이터 삭제 (마지막에 수행 필)
/usr/bin/python3.6 script_auto_clean_envs.py --m_target_index=$m_t --forward_ndx=$f_ndx


exit 0
