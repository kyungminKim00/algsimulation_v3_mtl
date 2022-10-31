import numpy as np

""" Declare static variables
"""
release = False
use_historical_model = True  # the historical best or the best model at the moment
re_assign_vars = True
b_select_model_batch = False  # For experimentals
dataset_version = None
forward_ndx = None
s_test = None
e_test = None
m_env = None
m_file_pattern = None
m_dataset_dir = None
n_off_batch = None
m_total_example = None  # it depends on a stride and data and random sampling strategies
timestep = None  # it depends on a stride and data and random sampling strategies
m_total_timesteps = (
    None  # it depends on a stride and data and random sampling strategies
)
m_final_model = None  # [None | 'fs_epoch_47_500'], only for the test stage
m_on_validation = None
predefined_fixed_lr = None
on_cloud = None
m_max_grad_norm = None
m_buffer_size = None
m_main_replay_start = None
m_target_index = None
target_name = None
m_name = None  # Caution: the directory would be deleted and then re-created
manual_vars_additional = True
b_activate = True  # check status before running the model selection script
r_model_cnt = 7  # at least 10 models are required to run the model selection script
pkexample_type = {
    "decoder": "pkexample_type_B",
    "num_features_1": 15,
    "num_features_2": 17,
}  # {pkexample_type_A: 'Original', pkexample_type_B: 'Use var mask', pkexample_type_C: 'Enable WR'}

""" Agent parameter
"""
_debug_on = False
_full_tensorboard_log = False
disable_derived_vars = False


# data set related
objective = "IF"  # ['IF' | 'FS' | 'MT']
file_data_vars = "./datasets/rawdata/index_data/data_vars_"
l_objective = str(objective).lower()
tf_record_location = ""
if objective == "IF":
    tf_record_location = "index_forecasting"
elif objective == "FS":
    tf_record_location = "fund_selection"
elif objective == "MT":
    tf_record_location = "market_timing"

blind_set_seq = 500

# agents related
m_cv_number = 0
m_inference_buffer = 6
m_n_cpu = 1
m_n_step = 7  # 20 -> 10 -> 7
m_verbose = 1
m_warm_up_4_inference = int(m_inference_buffer)
m_augmented_sample = int(40 / 4)  # (40samples / 4strides) = 10samples(2 months)
m_augmented_sample_iter = 5  # 3 times
m_tensorboard_log_update = 200
m_tabular_log_interval = 1
gn_alpha = 0.12
m_bound_estimation = False  # only for the test
m_bound_estimation_y = False  # only for the test, band with y prediction (could be configured at script_xx_test.py as well)
dynamic_lr = False
dynamic_coe = False
grad_norm = False
market_timing = False
weighted_random_sample = False
enable_non_shared_part = False
enable_lstm = True
default_net = "inception_resnet_v2_Dummy"
c_epoch = 600
derived_vars_th = {0: "094", 1: 0.94}
buffer_drop_rate = 1  # 0.05 -> 0.1
# (total_samples * 0.1) / buffer_drop_rate / train_batch_size * target_epoch
warm_up_update = 100  # (1425 * 0.1) / 0.1 / 32 * 5
cosine_lr = True


""" Model learning
"""
m_l2_norm = 0.0  # 0.00004 (inception default)
m_l1_norm = 0.0
m_batch_decay = 0.9997  # 0.9997 (inception default)
m_batch_epsilon = 0.001  # 0.001 (inception default)
m_drop_out = 0.8
m_vf_coef = 0.25
m_vf_coef_2 = 0.25
m_pi_coef = 0.1
m_ent_coef = 0.01
m_factor = 0.05
m_h_factor = 0.15  # it may need extra model
m_cov_factor = 0  # it may need extra model
m_learning_rate = 5e-3  # default start convergence from 450 updates
m_offline_learning_rate = 5e-4  # fixed learning rate for offline learning
m_min_learning_rate = 7e-6  # tuning  7e-5 -> 7e-7  -> 7e-6
m_lstm_hidden = 1  # 1 for 256
m_num_features = 1  # 1 for 512
cyclic_lr_min = float(2e-4 * 2.5)
cyclic_lr_max = float(2e-4 * 4)


""" Memory and simulation
"""
# m_buffer_size = int(m_total_example*m_n_cpu*m_n_step*0.05)  # real buffer size per buffer batch file
m_validation_interval = 30  # a validation is performed with every buffer & model drops and validation intervals
m_validation_min_epoch = 1
m_replay_ratio = 4
m_replay_start = m_n_cpu * 20
m_discount_factor = 5
# m_main_replay_start = int(m_total_example*1000000)  # actually disabled
# m_main_replay_start = int(m_total_example*0.99)

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
# m_interval = [15, 30, 45, 60, 75]  # sampling
# m_interval_value = [0.3, 0.2, 0, np.inf, -np.inf]  # sampling
m_early_stop = 15
m_interval = [2, 150]  # sampling
m_interval_value = [np.inf, -np.inf]  # sampling
m_forget_experience = 0.05
m_forget_experience_short_term = 0.5
improve_th = 0.25
m_replay_iteration = 1  # use when simulator meets saddle point
m_sample_th = 0.2


"""Train mode configuration
"""
m_online_buffer = False
search_variables = False
search_parameter = False
m_offline_buffer_file = ""
m_offline_learning_epoch = 0
m_sub_epoch = 0
m_pool_sample_num_test = 0
m_pool_sample_num = 0
m_pool_sample_ahead = 0
m_pool_corr_th = 0.7
m_mask_corr_th = 0.5
explane_th = 0.5
m_pool_sample_start = -(
    m_pool_sample_num_test + m_pool_sample_ahead + m_pool_sample_num
)
m_pool_sample_end = -1

m_train_mode = 0  # [0 | 1] 0: init train, 1: Transfer
m_pre_train_model = "./save/model/rllearn/IF_Gra05_FOb3_MultiR_LR10E4_ICD_Gold_Buffer/20200205_1856_fs_epoch_4_0_pe1.53_pl1.43_vl1.15_ev0.775.pkl"
if m_train_mode == 1:
    m_name = m_name + "_continued"


"""Declare functions
"""


def init_var(args):
    m_target_index = args.m_target_index
    target_name = target_id2name(m_target_index)
    m_name = objective + "_" + target_name
    return m_target_index, target_name, m_name


def target_id2name(m_target_index):
    target_name = ""
    if m_target_index == 0:
        target_name = "INX"
    elif m_target_index == 1:
        target_name = "KS"
    elif m_target_index == 2:
        target_name = "Gold"
    elif m_target_index == 3:
        target_name = "US10YT"
    elif m_target_index == 4:
        target_name = "FTSE"
    elif m_target_index == 5:
        target_name = "GDAXI"
    elif m_target_index == 6:
        target_name = "SSEC"
    elif m_target_index == 7:
        target_name = "BVSP"
    elif m_target_index == 8:
        target_name = "N225"
    elif m_target_index == 9:
        target_name = "GB10YT"
    elif m_target_index == 10:
        target_name = "DE10YT"
    elif m_target_index == 11:
        target_name = "KR10YT"
    elif m_target_index == 12:
        target_name = "CN10YT"
    elif m_target_index == 13:
        target_name = "JP10YT"
    elif m_target_index == 14:
        target_name = "BR10YT"
    return target_name


def get_file_name(m_target_index, file_data_vars):
    return file_data_vars + target_id2name(m_target_index) + "_intermediate.csv"


assert not (objective == "MT" and market_timing), "check environment setting"

if _debug_on:
    img_jpeg = {
        "width": 1860,
        "height": 980,
        "dpi": 600,
    }  # full_jpeg = {'width': 1860, 'height': 980]
else:
    img_jpeg = {
        "width": 18,
        "height": 10,
        "dpi": 10,
    }  # full_jpeg = {'width': 1860, 'height': 980]
