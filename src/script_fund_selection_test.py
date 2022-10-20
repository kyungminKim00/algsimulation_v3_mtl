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
import numpy as np
import pandas as pd
import header.fund_selection.RUNHEADER as RUNHEADER
from datasets.fund_selection_protobuf2pickle import DataSet

# from custom_model.policies.policies_fund_selection import MlpPolicy, MlpLstmPolicy, \
#    MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from custom_model.fund_selection.common import SubprocVecEnv, set_global_seeds

# from custom_model import ACER
from custom_model.fund_selection import A2C

import os
import pickle
import sys

import matplotlib

# matplotlib.use('cairo')
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class Script:
    def __init__(self, so=None):
        self.so = so

    def run(self, mode, env_name=None, tensorboard_log=None,
            full_tensorboard_log=None, model_location=None,
            n_cpu=None, n_step=None, total_timesteps=None, result=None, m_inference_buffer=None):

        env = SubprocVecEnv([lambda: util.make(env_name) for i in range(n_cpu)])
        env.set_attr('so', self.so)
        env.set_attr('mode', mode)

        # get model list for evaluate performance
        filenames = list()
        models = os.listdir(model_location)
        [filenames.append(_model) for _model in models if '.pkl' and 'fs' in _model and int(_model.split('_')[4]) > 50]
        filenames.sort()

        performance_track = list()
        for _model in filenames:
            """Inference
            """
            current_step = 0
            env.set_attr('current_step', current_step)
            env.set_attr('eof', False)
            env.set_attr('steps_beyond_done', None)
            env.env_method('clear_cache')
            obs = env.reset()

            _model = '{}/{}'.format(model_location, _model)
            print('\n\nloading model: {} '.format(_model))
            model = A2C.load(_model)
            tmp_info, selected_fund, img_action, act_selection = list(), list(), list(), list()
            p_states, p_dones, action, info = None, None, None, None
            while np.sum(np.array(env.get_attr('eof'))) == 0:
                # p_states, p_dones, action, info = None, None, None, None

                action, states, values, neglogp = model.predict(obs, state=p_states,
                                                                mask=p_dones, deterministic=True)
                action_pro = model.action_probability(obs, state=p_states, mask=p_dones, actions=action)
                p_states = states
                _, rewards, dones, info = env.step(action)

                tmp = np.array([[returns['date'], cent, apro, returns['0day_return']] for cent, apro, returns in
                                zip(neglogp, action_pro.T[0].tolist(), info)])

                # for sub_current_step in range(m_inference_buffer):
                #     action, states, values, neglogp = model.predict(obs, state=p_states,
                #                                                     mask=p_dones, deterministic=True)
                #     action_pro = model.action_probability(obs, state=p_states, mask=p_dones, actions=action)
                #     p_states = states
                #
                #     env.set_attr('current_step', current_step + sub_current_step + 1)
                #     if (sub_current_step + 1) < m_inference_buffer:
                #         obs, rewards, dones, _ = env.step(action)
                #         p_dones = dones
                #     else:
                #         _, _, _, info = env.step(action)
                #         tmp = np.array([[returns['date'], cent, apro, returns['5day_return']] for cent, apro, returns in
                #                         zip(neglogp, action_pro.T[0].tolist(), info)])
                #         # con1 = [[cent, idx] for cent, idx in zip(neglogp, range(len(neglogp)))]
                #         # con2 = [[apro, idx] for apro, idx in zip(action_pro.T[0].tolist(), range(len(action_pro)))]
                #         # con3 = [[returns['5day_return'], idx] for returns, idx in zip(info, range(len(info)))]
                #         #
                #         # con1.sort()
                #         # con2.sort(reverse=True)
                #         # con3.sort(reverse=True)

                act_selection.append(tmp.tolist())
                img_action.append(action[0])

                current_step = current_step + 1
                env.set_attr('current_step', current_step)
                obs, _, _, _ = env.step(action)

                for idx in range(len(info)):
                    tmp_info.append([info[idx]['date'], info[idx]['0day_return'], info[idx]['m5day_return'],
                                     info[idx]['5day_return'], info[idx]['10day_return'],
                                     info[idx]['20day_return'], info[idx]['60day_cov'],
                                     len(info[idx]['selected_fund_name']), info[idx]['selected_fund_name']])
                    selected_fund.append([info[idx]['selected_fund_name'], info[idx]['selected_action']])
                    sys.stdout.write('\r>> Test Date:  %s' % info[idx]['date'])
                    sys.stdout.flush()

            """File out information
            """
            tmp_info = np.array(tmp_info, dtype=np.object)
            summary = tmp_info[:, 1:7]
            hit_summary = summary
            act_selection = np.array(act_selection, dtype=np.object)
            act_selection = np.reshape(act_selection,
                                       (act_selection.shape[0] * act_selection.shape[1], act_selection.shape[2]))

            # s_test = np.argwhere(tmp_info[:, 0] == RUNHEADER.m_s_test)[-1][0] + 1
            # e_test = np.argwhere(tmp_info[:, 0] == RUNHEADER.m_e_test)[-1][0] + 1
            # tmp_info = tmp_info[s_test:e_test]
            # act_selection = act_selection[s_test:e_test]

            summary = (np.sum(summary, axis=0)) / n_cpu
            hit_summary = np.where(hit_summary < 0, 0, 1)
            hit_summary = np.sum(hit_summary, axis=0) / len(tmp_info)
            num_funds = int((np.sum(tmp_info[:, 7], axis=0)) / len(tmp_info))
            df = pd.DataFrame(data=tmp_info, columns=['date', '0day_return', 'm5day_return', '5day_return', \
                                                      '10day_return', '20day_return', '60day_cov', \
                                                      'num_funds', 'selected_fund_name'])
            current_model = _model.split('/')[-1]
            current_model = current_model.split('.pkl')[0]
            prefix = result + '/test_' + current_model
            df.to_csv(prefix + '___{:3.2}___0_{:3.2}_1_{:3.2}_5_{:3.2}_10_{:3.2}_20_{:3.2}_{}_{}.csv'. \
                      format(summary[0], hit_summary[0], hit_summary[1], hit_summary[2], hit_summary[3], hit_summary[4],
                             num_funds, model_location.split('/')[-1]))

            # df = pd.DataFrame(data=act_selection, columns=['date', 'c_ent', 'a_pro', '0day_return'])
            # df.to_csv(prefix + '___{:3.2}__{}_{}.csv'.format(summary[0], num_funds, model_location.split('/')[-1]))

            # tmp = np.reshape(action, [self.n_envs, self.n_steps, actions.shape[-1]])[0].T * 255
            # plt.imsave('./save/images/{}_{}.jpeg'.format(str(delete_later), update), tmp)

            tmp = np.array(img_action).T * 255
            plt.imsave('{}/{}.jpeg'.format(result, current_model), tmp)

            # tmp = np.abs(np.diff(tmp, axis=1))
            # plt.imsave('{}/{}_diff_{}.jpeg'.format(result, current_model, int(np.sum(tmp))), tmp)

            # # Disable for now
            # performance_track.append([current_model.split('_')[1], summary[2], hit_summary[0], hit_summary[1],
            #                           hit_summary[2], hit_summary[3], hit_summary[4]])

        # df = pd.DataFrame(data=np.array(performance_track),
        #                   columns=['Time_Step', 'return', '0day_return', '1day_return', '5day_return', '10day_return',
        #                            '20day_return'])
        # df.to_csv(result + '/performance_track.csv')


if __name__ == '__main__':
    """configuration
    """
    m_name = RUNHEADER.m_name
    m_inference_buffer = RUNHEADER.m_inference_buffer

    _model_location = './save/model/rllearn/' + m_name
    _tensorboard_log = './save/tensorlog/fund_selection/' + m_name
    _mode = 'test'  # [train | validation | test]
    _dataset_dir = RUNHEADER.m_dataset_dir
    _full_tensorboard_log = False
    _result = './save/result'
    _result = '{}/{}'.format(_result, _model_location.split('/')[-1])

    # load meta
    with open(_model_location + '/meta', mode='rb') as fp:
        meta = pickle.load(fp)
        fp.close()

    _n_step = meta['_n_step']
    _cv_number = meta['_cv_number']
    _n_cpu = meta['_n_cpu']
    _env_name = meta['_env_name']
    _file_pattern = meta['_file_pattern']

    """ run application
    """
    # register
    BaseManager.register('DataSet', DataSet)
    manager = BaseManager()
    manager.start()

    # dataset injection
    sc = Script(so=manager.DataSet(
        dataset_dir=_dataset_dir, file_pattern=_file_pattern,
        split_name=_mode, cv_number=_cv_number))

    sc.run(mode=_mode, env_name=_env_name, tensorboard_log=_tensorboard_log,
           full_tensorboard_log=_full_tensorboard_log,
           model_location=_model_location,
           n_cpu=_n_cpu, n_step=_n_step, result=_result, m_inference_buffer=m_inference_buffer)

    # print test environments
    print('\nEnvs ID: {}'.format(_env_name))
    print('Data Set Number: {}'.format(_cv_number))
    print('Num Agents: {}'.format(_n_cpu))
    print('Num Step: {}'.format(_n_step))
    print('Result Directory: {}'.format(_result))
