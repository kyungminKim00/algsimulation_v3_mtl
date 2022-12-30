#!/bin/bash

# open with binary. make sure file format is unix

echo "Change file permisstion"
chmod -R 777 .


p_id=100
d_v=v10
m_t=3

python3 script_index_forecasting_train.py --m_online_buffer=1 --search_parameter=1 --process_id=$p_id --on_cloud=0 --n_cpu=3 --dataset_version=$d_v --m_target_index=$m_t
for k in 1 2 1 2 1 1
do
  for i in $p_id
  do
    sp=$k
    idx=$i
    echo "python3 script_index_forecasting_train.py --m_online_buffer=0 --search_parameter=$k --process_id=$idx --on_cloud=1 --n_cpu=0 > train_pid$idx.out &"
    python3 script_index_forecasting_train.py --m_online_buffer=0 --search_parameter=$k --process_id=$idx --on_cloud=0 --n_cpu=0 --m_target_index=$m_t

    echo "python3 script_index_forecasting_test.py --process_id=$idx > test_pid$idx.out &"
    python3 script_index_forecasting_test.py --process_id=$idx

    echo "python3 auto_clean_envs.py --process_id=$idx > pid$idx.out &"
    python3 auto_clean_envs.py --process_id=$idx
  done
done

exit 0
