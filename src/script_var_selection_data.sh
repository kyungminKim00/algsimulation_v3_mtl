#!/bin/bash

# open with binary. make sure file format is unix e.g vim -b filename
# :%s/^M//g (^M=Ctrl+v+m)

s_str='var_select_factor = 0.1'
t_str='var_select_factor = 0.2'
sed -ri 's/$s_str $t_str/' ./header/RUNHEADER.py

python script_index_forecasting_generate_data.py --e_test="2010-01-01" --dataset_version="v0" --m_target_index=8 --gen_var=0 --domain=None