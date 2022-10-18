import re
import os
import sys
from PIL import Image
import glob
import platform
import numpy as np
import header.index_forecasting.RUNHEADER as RUNHEADER
import argparse
import datetime
import util
import pickle
from util import get_domain_on_CDSW_env
from datasets.windowing import (
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
    fun_mean,
    fun_cumsum,
    fun_cov,
    fun_cross_cov,
)

import matplotlib

matplotlib.use("Agg")  # Bypass the need to install Tkinter GUI framework

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil

# model_repo_meta information
base_dir = './save/model_repo_meta'
base_dir2 = './save/model/rllearn'
stored_models = list()
for it in os.listdir(base_dir):
    fp = open(base_dir + '/' + it, 'rb')
    model_dict = pickle.load(fp)
    fp.close()

    for item in model_dict:
        if item['model_name'] is not None:
            tmp_dir = '/'.join([base_dir2, item['m_name']])
            tmp_file = item['model_name']
            s_file = '/'.join([tmp_dir, tmp_file])

            stored_models.append([tmp_dir, tmp_file])
            b_test = os.path.isfile(s_file)
            if not b_test:
                print('File Not exist: {}: {}'.format(s_file, b_test))


# delete files
for base_dir, model in stored_models:
    try:
        for c_model in os.listdir(base_dir):
            if '.pkl' in c_model and c_model != model:
                os.remove('/'.join([base_dir, c_model]))
                print('Delete dummy files: {}'.format('/'.join([base_dir, c_model])))
    except FileNotFoundError:
        pass

# delete folders
for t_dir in os.listdir(base_dir2):
    delete = True
    if t_dir == 'buffer_save':
        delete = False
    else:
        for ch in stored_models:
            if t_dir == ch[0].split('/')[-1]:
                delete = False
    
    if delete:
        shutil.rmtree('/'.join([base_dir2, t_dir]), ignore_errors=True)
        print('Delete dummy folder {} : {}'.format(base_dir2, t_dir))
    

    