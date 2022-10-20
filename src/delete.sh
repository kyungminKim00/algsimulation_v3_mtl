#!/bin/bash

f_ndx=20
m_t=0
ds_v=$(($m_t+11))

# 모형 평가 7번 병렬 수행
for k in 0 1 2 3 
do
        /usr/bin/python3.6 script_index_forecasting_test.py --process_id=$(($main_pid+$k)) --operation_mode=0
done



# 모형 선택
/usr/bin/python3.6 script_index_forecasting_select_model.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx

# 후 처리
/usr/bin/python3.6 script_index_forecasting_adhoc.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=1

# 최종 예측 값 생성
/usr/bin/python3.6 script_index_forecasting_test.py --dataset_version=v$ds_v --m_target_index=$m_t --forward_ndx=$f_ndx --operation_mode=1

:<<"END"
# 임시 생성 데이터 삭제 (마지막에 수행 필)
/usr/bin/python3.6 script_auto_clean_envs.py --m_target_index=$m_t --forward_ndx=$f_ndx
END


exit 0
