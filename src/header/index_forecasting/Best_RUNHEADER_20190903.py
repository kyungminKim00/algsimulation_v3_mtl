import numpy as np

""" Agent parameter
"""
# data set
dataset_version = 'v0'
m_env = 'IF-' + dataset_version
m_file_pattern = 'if_' + dataset_version + '_cv%02d_%s.pkl'  # data set file name
m_dataset_dir = './save/tf_record/index_forecasting/if_x0_20_y20_' + dataset_version  # data set location
m_cv_number = 0

# agent opt
m_name = 'IF_onPiceof001_GraClip_FOb_ICD'  # Caution: at first, this directory would be deleted and then re-created
m_target_index = 0  # check y_index.txt {0: INX, 1:KS200}
# m_name = 'Delete_20190716'
m_inference_buffer = 20
# m_n_cpu = (mp.cpu_count()-2)*2
m_n_cpu = 20
# m_n_cpu = 2
m_n_step = 20
# m_n_step = 5
m_verbose = 1
m_total_example = 571  # 1825 cause memory error for full observation
timestep = m_n_step*m_n_cpu*m_total_example
m_total_timesteps = timestep*2
m_tensorboard_log_update = 10
m_tabular_log_interval = 1
# m_buffer_drop = int(m_total_example*0.1*m_n_cpu)  # not in use
# m_buffer_drop = 2  # not in use
m_s_test = '2017-04-09'  # disabled for now
m_e_test = '2018-11-19'  # disabled for now
# m_s_test = '2013-07-22'
# m_e_test = '2014-06-30'
# {nature_cnn [n_env by 5X20 by n_index], inception_resnet_v2 [n_env by 299X299 by n_index]}
# [ 'nature_cnn_A' | 'nature_cnn_B' | 'inception_resnet_v2_A' | 'inception_resnet_v2_B' | 'inception_resnet_v2_C' | ...]
default_net = 'inception_resnet_v2_D'
m_bound_estimation = False  # only for test parameter
m_final_model = None  # [None | 'fs_epoch_47_500'], only for test parameter
m_max_grad_norm = 0.5  # default 0.5 and None for Inception net

""" Model learning parameter
"""
# m_vf_coef = 0.25
# m_ent_coef = 0.01
# m_vf_coef = 0.15
# m_vf_coef = 1  # Cov_20190619_DropBuff
# m_ent_coef = 0.01  # Cov_20190619_DropBuff
m_vf_coef = 1  # Cov_20190619_DropBuff2, default: 0.2
# m_ent_coef = 0.01  # Cov_20190619_DropBuff2
m_ent_coef = 0  # Cov_20190715_onBFlearn_onState_v2
m_pi_coef = 0.01
# m_ent_coef = 0  # Cov_20190619_DropBuff2
# m_h_factor = 0.125  # first
# m_factor = 0.125  # first
# m_h_factor = 0.05  # default
# m_h_factor = 0.15  # increase
# m_factor = 0.05  # default
m_factor = 0.05
m_h_factor = 0.15  # it may need extra model
# m_cov_factor = 2  # it may need extra model
m_cov_factor = 0  # it may need extra model
# m_learning_rate = 16e-6  # too slow.. start convergence from 600 updates (240000 time steps)
# m_learning_rate = 5e-4  # default start convergence from 450 updates
m_learning_rate = 5e-13  # test
# m_learning_rate = 5e-3
# m_learning_rate = 125e-5  # too fast convergence
# m_offline_learning_rate = 5e-6  # fixed learning rate for offline learning
m_offline_learning_rate = 5e-3  # fixed learning rate for offline learning
# m_hidden = 5
m_lstm_hidden = 1  # 1 for 256
m_num_features = 1  # 1 for 512

""" Memory and simulation parameter
"""
m_buffer_size = int(m_total_example*0.2*m_n_cpu*m_n_step)  # real buffer size per buffer batch file
m_replay_ratio = 4
m_replay_start = m_n_cpu*20
# m_main_replay_start = int(m_total_example*1000000)  # actually disable
m_main_replay_start = int(m_total_example*0.7)
m_discount_factor = 5

# short-term threshold ((m_entry_th or m_mask_th) and (hit_ratio and num_of_action))
m_entry_th = 10  # init threshold
m_min_entry_th = 3  # min bound
m_max_entry_th = 40  # max bound
m_mask_th = 2  # init threshold
m_min_mask_th = 1  # min bound
m_max_mask_th = 15  # max bound
# m_entry_th = 18  # init threshold
# m_min_entry_th = 15  # min bound
# m_max_entry_th = 30  # max bound
# m_mask_th = 4  # init threshold
# m_min_mask_th = 3  # min bound
# m_max_mask_th = m_n_step*0.75  # max bound

# 144 setting (num_fund)
m_max_iter = 1  # adjust threshold
m_exit_cnt = 3  # max search space (m_max_iter*m_exit_cnt)
m_target_actions = 3
m_allow_actions_min = 0
m_allow_actions_max = 5

# # 30 setting (num_fund)
# m_max_iter = 1  # adjust threshold
# m_exit_cnt = 3  # max search space (m_max_iter*m_exit_cnt)
# m_target_actions = 9
# m_allow_actions_min = 3
# m_allow_actions_max = 15

# m_allow_actions_max = 50
m_early_stop = 15
# m_interval = [15, 30, 45, 60, 75]  # sampling
# m_interval_value = [0.3, 0.2, 0, np.inf, -np.inf]  # sampling
m_interval = [2, 150]  # sampling
m_interval_value = [np.inf, -np.inf]  # sampling
m_forget_experience = 0.05
m_forget_experience_short_term = 0.5
improve_th = 0.25
m_replay_iteration = 1  # use when simulator meets saddle point
m_sample_th = 0.2

"""Train mode configuration
"""
# mode 1. using m_pre_train_model
m_train_mode = 0  # [0 | 1] 0: init train, 1: model load and train
m_online_buffer = False  # [True | False]
m_pre_train_model = './save/model/rllearn/IF_v0_SP_VL1_VnraNet_NGraClip_FOb_NCA_LR_Buffer/fs_epoch_5_0_pe1.9_pl1.79_vl1.17.pkl'
m_offline_buffer_file = './save/model/rllearn/buffer_save/IF_v0_SP_VL1_VnraNet_NGraClip_FOb'
if m_train_mode == 1:
    m_name = m_name + '_continued'
# mode 2. using buffer mode
if not m_online_buffer:
    m_train_mode = 0
    m_offline_learning_epoch = 1000
    timestep = m_buffer_size * m_offline_learning_epoch
    m_name = m_name + '_Buffer'
