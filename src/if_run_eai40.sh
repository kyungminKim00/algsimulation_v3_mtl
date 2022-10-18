#!/bin/bash

# open with binary. make sure file format is unix

echo "Change file permisstion"
chmod -R 777 .

python3 script_generate_data.py --dataset_version=v10 --s_test=2016-07-04 --e_test=2016-10-03 --verbose=3 --m_target_index=3

exit 0
