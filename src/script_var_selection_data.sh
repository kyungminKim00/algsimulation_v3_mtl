#!/bin/bash

sed -ri 's/var_select_factor = 0.3/var_select_factor = 0.5/' ./header/index_forecasting/RUNHEADER.py
python script_index_forecasting_generate_data.py --e_test="2010-01-01" --dataset_version="v0" --m_target_index=8 --gen_var=0 --domain=None
