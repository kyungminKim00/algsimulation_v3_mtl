from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import util
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import numpy as np
import pandas as pd

from custom_model.policies.policies_fund_selection import MlpPolicy, MlpLstmPolicy, \
    MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from custom_model.common import SubprocVecEnv, set_global_seeds

from custom_model import ACER
from custom_model import A2C


class Script:
    def __init__(self, so=None):
        self.so = so

    def run(self, mode, env_name=None, tensorboard_log=None,
            full_tensorboard_log=None, model_location=None,
            n_cpu=None, n_step=None, total_timesteps=None):

        env = SubprocVecEnv([lambda: util.make(env_name) for i in range(n_cpu)])
        env.set_attr('so', self.so)
        env.set_attr('mode', mode)

        if mode == 'train':
            # env.reset()
            # env.env_method('method_test', 111)

            # MultiDiscrete, Continuous not implemented yet for ACER
            # model = ACER(CnnLnLstmPolicy, env, verbose=1, n_steps=self.n_step,
            #              tensorboard_log=self.tensorboard_log, full_tensorboard_log=self.full_tensorboard_log)
            model = A2C(CnnLnLstmPolicy, env, verbose=1, n_steps=n_step,
                        tensorboard_log=tensorboard_log, full_tensorboard_log=full_tensorboard_log)
            model.learn(total_timesteps=total_timesteps, seed=0)
            model.save(model_location)
            print('model is saved ')

            del model  # remove to demonstrate saving and loading
        else:
            print('loading model ')
            model = A2C.load(model_location)
            print('done...!!!')

            obs = env.reset()
            tmp_info = list()
            selected_fund = list()

            while np.sum(np.array(env.get_attr('eof'))) == 0:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

                for idx in range(len(info)):
                    print('date: {}, today_return: {}, next_return: {}, 5day_return: {}, 10day_return: {}, '
                          '20day_return: {}'. \
                          format(info[idx]['date'], info[idx]['0day_return'],
                                 info[idx]['1day_return'], info[idx]['5day_return'],
                                 info[idx]['10day_return'], info[idx]['20day_return']))
                    tmp_info.append([info[idx]['date'],info[idx]['0day_return'],info[idx]['1day_return'],
                                info[idx]['5day_return'],info[idx]['10day_return'],
                                info[idx]['20day_return'], info[idx]['selected_fund_name']])
                    selected_fund.append([info[idx]['selected_fund_name'], info[idx]['selected_action']])
            pd.DataFrame(data=np.array(tmp_info), columns=['date', '0day_return', '1day_return', '5day_return', \
                                                           '10day_return', '20day_return', 'selected_fund_name']).\
                to_csv('./save/result/test_fs_info_{}.csv'.format(model_location.split('/')[-1]))
            pd.DataFrame(np.array(selected_fund)).to_csv('./save/result/test_fs_fundlist_{}.csv'.format(model_location.split('/')[-1]))



if __name__ == '__main__':
    """configuration
    """
    _env_name = 'FS-v0'
    _tensorboard_log = './save/tensorlog/fund_selection'
    _full_tensorboard_log = False
    _dataset_dir = './save/tf_record/fund_selection'
    _model_location = './save/model/rllearn/fund_selection_1'
    _file_pattern = 'fs_v0_cv%02d_%s.pkl'
    _n_step = 5
    _cv_number = 0
    _mode = 'train'  # [train | validation | test]
    _n_cpu = 7
    # _n_cpu = 2
    _total_timesteps = 1320000000000

    """ run application
    """
    # register
    from datasets.protobuf2pickle import DataSet

    BaseManager.register('DataSet', DataSet)
    manager = BaseManager()
    manager.start()

    # dataset injection
    if _mode == 'train':
        sc = Script(so=manager.DataSet(
            dataset_dir=_dataset_dir, file_pattern=_file_pattern,
            split_name=_mode, cv_number=_cv_number, n_batch_size=_n_step))
    else:
        sc = Script(so=manager.DataSet(
            dataset_dir=_dataset_dir, file_pattern=_file_pattern,
            split_name=_mode, cv_number=_cv_number, n_batch_size=_n_step))

    sc.run(mode=_mode, env_name=_env_name, tensorboard_log=_tensorboard_log,
           full_tensorboard_log=_full_tensorboard_log,
           model_location=_model_location,
           n_cpu=_n_cpu, n_step=_n_step,
           total_timesteps=_total_timesteps)
