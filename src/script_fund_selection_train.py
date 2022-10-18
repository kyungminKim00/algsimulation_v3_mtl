from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import util
# import multiprocessing as mp
from multiprocessing.managers import BaseManager
# import numpy as np
# import pandas as pd
import shutil
import header.fund_selection.RUNHEADER as RUNHEADER
from datasets.fund_selection_protobuf2pickle import DataSet

from custom_model.fund_selection.policies.policies_fund_selection import MlpPolicy, MlpLstmPolicy, \
    MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from custom_model.fund_selection.common import SubprocVecEnv, set_global_seeds

# from custom_model.fund_selection import ACER
from custom_model.fund_selection import A2C
import os
import pickle

# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt

class Script:
    def __init__(self, so=None):
        self.so = so

    def run(self, mode, env_name=None, tensorboard_log=None, verbose=None,
            full_tensorboard_log=None, model_location=None, learning_rate=None,
            n_cpu=None, n_step=None, total_timesteps=None, log_interval=None):

        env = SubprocVecEnv([lambda: util.make(env_name) for i in range(n_cpu)])

        # init environments
        init_idx = 0
        env.set_attr('so', self.so)
        env.set_attr('mode', mode)
        env.set_attr('current_episode_idx', 0)
        env.set_attr('current_step', 0)
        if RUNHEADER.m_train_mode == 0:
            model = A2C(CnnLnLstmPolicy, env, verbose=verbose, n_steps=n_step, learning_rate=learning_rate,
                        tensorboard_log=tensorboard_log, full_tensorboard_log=full_tensorboard_log,
                        policy_kwargs={'n_lstm': int(256*RUNHEADER.m_lstm_hidden), 'is_training': True})
        else:  # use pre-trained model
            print('\nloading model ')
            model = A2C.load(RUNHEADER.m_pre_train_model, env)

        model.learn(total_timesteps=total_timesteps, seed=0, model_location=model_location, log_interval=log_interval)
        model.save(model_location)
        del model  # remove to demonstrate saving and loading


if __name__ == '__main__':
    """configuration
    """
    m_name = RUNHEADER.m_name
    _n_step = RUNHEADER.m_n_step
    _n_cpu = RUNHEADER.m_n_cpu
    _total_timesteps = RUNHEADER.m_total_timesteps
    _learning_rate = RUNHEADER.m_learning_rate
    _verbose = RUNHEADER.m_verbose
    _log_interval = RUNHEADER.m_tabular_log_interval

    _model_location = './save/model/rllearn/'+m_name
    _tensorboard_log = './save/tensorlog/fund_selection/'+m_name

    # mkdir for model, log, and result
    for k in range(3):
        if k == 0:
            target = _model_location
        elif k == 1:
            target = _tensorboard_log
        elif k == 2:
            location = _model_location.split('/')
            target = '{}/{}/result/{}'.format(location[0], location[1], location[4])

        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        os.mkdir(target)

    _mode = 'train'  # [train | validation | test]
    _full_tensorboard_log = True
    _dataset_dir = RUNHEADER.m_dataset_dir

    _env_name = RUNHEADER.m_env
    _file_pattern = RUNHEADER.m_file_pattern
    _cv_number = RUNHEADER.m_cv_number

    meta = {
        '_env_name': _env_name, '_model_location': _model_location, '_file_pattern': _file_pattern,
        '_n_step': _n_step, '_cv_number': _cv_number, '_n_cpu': _n_cpu
    }

    # save number of environments
    with open(_model_location + '/meta', mode='wb') as fp:
        pickle.dump(meta, fp)
        fp.close()


    """ run application
    """
    BaseManager.register('DataSet', DataSet)
    manager = BaseManager()
    manager.start()

    # dataset injection
    sc = Script(so=manager.DataSet(
        dataset_dir=_dataset_dir, file_pattern=_file_pattern,
        split_name=_mode, cv_number=_cv_number, n_batch_size=_n_step))

    sc.run(mode=_mode, env_name=_env_name, tensorboard_log=_tensorboard_log,
           full_tensorboard_log=_full_tensorboard_log,
           model_location=_model_location, verbose=_verbose,
           n_cpu=_n_cpu, n_step=_n_step, learning_rate=_learning_rate,
           total_timesteps=_total_timesteps, log_interval=_log_interval)
