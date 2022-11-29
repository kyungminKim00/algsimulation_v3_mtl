from typing import Any, Dict, List, Literal, Tuple

import numpy as np

""" Agent parameter
"""
_debug_on: bool = True
full_tensorboard_log: bool = False
disable_derived_vars: bool = True
m_on_validation: bool = False

""" Declare static variables
"""
var_select_factor = (
    0.4  # 0 < vsf < 1, select similar assets against the target as close as 0
)
m_pool_samples = 70
release: bool = False
use_historical_model: bool = True  # the historical best or the best model at the moment
re_assign_vars: bool = True  # add forced variables
manual_vars_additional: bool = True  # add forced variables
b_select_model_batch: bool = False  # For experimentals
dataset_version: str = None
forward_ndx: int = None
s_test: str = None
e_test: str = None
m_env: str = None
m_file_pattern: str = None
m_dataset_dir: str = None
n_off_batch: int = None
m_total_example: int = (
    None  # it depends on a stride and data and random sampling strategies
)
timestep: int = None  # it depends on a stride and data and random sampling strategies
m_total_timesteps: int = (
    None  # it depends on a stride and data and random sampling strategies
)
m_final_model: str = None  # [None | 'fs_epoch_47_500'], only for the test stage
predefined_fixed_lr: float = None
on_cloud: int = None
m_max_grad_norm: float = None
m_buffer_size: int = None
m_main_replay_start: int = None
m_target_index: int = None
target_name: str = None
m_name: str = None  # Caution: the directory would be deleted and then re-created
b_activate: bool = False  # check status before running the model selection script
r_model_cnt: int = (
    7  # at least 10 models are required to run the model selection script
)
pkexample_type: Dict[str, Any] = {
    "decoder": "pkexample_type_B",
    "num_features_1": 5,
    "num_features_2": 17,
    "num_market": 15,
}  # {pkexample_type_A: 'Original', pkexample_type_B: 'Use var mask', pkexample_type_C: 'Enable WR'}
raw_x: str
raw_x2: str
raw_y = "./datasets/rawdata/index_data/gold_index.csv"
predefined_std = "./datasets/rawdata/index_data/predefined_std_"
gen_var: int
use_c_name: str
use_var_mask: bool
max_x: int

# data set related
objective: str = "IF"  # ['IF' | 'FS' | 'MT']
file_data_vars: str = "./datasets/rawdata/index_data/data_vars_"
l_objective: str = str(objective).lower()
tf_record_location: str = ""
if objective == "IF":
    tf_record_location = "index_forecasting"
elif objective == "FS":
    tf_record_location = "fund_selection"
elif objective == "MT":
    tf_record_location = "market_timing"

blind_set_seq: int = 500

# agents related
m_cv_number: int = 0
m_inference_buffer: int = 6
m_n_cpu: int = 1
m_n_step: int = 7  # 20 -> 10 -> 7
m_verbose: int = 1
m_warm_up_4_inference: int = int(m_inference_buffer)
m_augmented_sample: int = int(40 / 4)  # (40samples / 4strides) = 10samples(2 months)
m_augmented_sample_iter: int = 5  # 3 times
m_tensorboard_log_update: int = 200
m_tabular_log_interval: int = 1
gn_alpha: float = 0.12
m_bound_estimation: bool = False  # only for the test
m_bound_estimation_y: bool = False  # only for the test, band with y prediction (could be configured at script_xx_test.py as well)
dynamic_lr: bool = False
dynamic_coe: bool = False
grad_norm: bool = False
market_timing: bool = False
weighted_random_sample: bool = False
enable_non_shared_part: bool = False
enable_lstm: bool = True
default_net: bool = "inception_resnet_v2_Dummy"
c_epoch: int = 0  # learning model drop epoch
derived_vars_th: Dict[int, float] = {0: "094", 1: 0.94}
buffer_drop_rate: bool = 1  # 0.05 -> 0.1
# (total_samples * 0.1) / buffer_drop_rate / train_batch_size * target_epoch
warm_up_update: bool = 100  # (1425 * 0.1) / 0.1 / 32 * 5
cosine_lr: bool = True
mtl_target = 15  # 15 markets

""" Model learning
"""
m_l2_norm: float = 0.0  # 0.00004 (inception default)
m_l1_norm: float = 0.0
m_batch_decay: float = 0.9997  # 0.9997 (inception default)
m_batch_epsilon: float = 0.001  # 0.001 (inception default)
m_drop_out: float = 0.8
m_vf_coef: float = 0
m_pi_coef: float = 0
m_ent_coef: float = 0
m_factor: float = 0.05
m_h_factor: float = 0.15  # it may need extra model
m_cov_factor: int = 0  # it may need extra model
m_learning_rate: float = 5e-3  # default start convergence from 450 updates
m_offline_learning_rate: float = 5e-4  # fixed learning rate for offline learning
m_min_learning_rate: float = 7e-6  # tuning  7e-5 -> 7e-7  -> 7e-6
m_lstm_hidden: int = 1  # 1 for 256
m_num_features: int = 1  # 1 for 512
cyclic_lr_min: float = float(2e-4 * 2.5)
cyclic_lr_max: float = float(2e-4 * 4)


""" Memory and simulation
"""
# m_buffer_size = int(m_total_example*m_n_cpu*m_n_step*0.05)  # real buffer size per buffer batch file
m_validation_interval: int = 30  # a validation is performed with every buffer & model drops and validation intervals
m_validation_min_epoch: int = 1
m_replay_ratio: int = 4
m_replay_start: int = m_n_cpu * 20
m_discount_factor: int = 5
# m_main_replay_start = int(m_total_example*1000000)  # actually disabled
# m_main_replay_start = int(m_total_example*0.99)

# short-term threshold ((m_entry_th or m_mask_th) and (hit_ratio and num_of_action))
m_entry_th: int = 10  # init threshold
m_min_entry_th: int = 3  # min bound
m_max_entry_th: int = 40  # max bound
m_mask_th: int = 2  # init threshold
m_min_mask_th: int = 1  # min bound
m_max_mask_th: int = 15  # max bound
# m_entry_th = 18  # init threshold
# m_min_entry_th = 15  # min bound
# m_max_entry_th = 30  # max bound
# m_mask_th = 4  # init threshold
# m_min_mask_th = 3  # min bound
# m_max_mask_th = m_n_step*0.75  # max bound

# 144 setting (num_fund)
m_max_iter: int = 1  # adjust threshold
m_exit_cnt: int = 3  # max search space (m_max_iter*m_exit_cnt)
m_target_actions: int = 3
m_allow_actions_min: int = 0
m_allow_actions_max: int = 5

# # 30 setting (num_fund)
# m_max_iter = 1  # adjust threshold
# m_exit_cnt = 3  # max search space (m_max_iter*m_exit_cnt)
# m_target_actions = 9
# m_allow_actions_min = 3
# m_allow_actions_max = 15

# m_allow_actions_max = 50
# m_interval = [15, 30, 45, 60, 75]  # sampling
# m_interval_value = [0.3, 0.2, 0, np.inf, -np.inf]  # sampling
m_early_stop: int = 15
m_interval: List[int] = [2, 150]  # sampling
m_interval_value: List[float] = [np.inf, -np.inf]  # sampling
m_forget_experience: float = 0.05
m_forget_experience_short_term: float = 0.5
improve_th: float = 0.25
m_replay_iteration: int = 1  # use when simulator meets saddle point
m_sample_th: float = 0.2


"""Train mode configuration
"""
m_online_buffer: bool = False
search_variables: bool = False
search_parameter: bool = False
m_offline_buffer_file: str = ""
m_offline_learning_epoch: int = 0
m_sub_epoch: int = 0
m_pool_sample_num_test: int = 0
m_pool_sample_num: int = 0
m_pool_sample_ahead: int = 0
m_pool_corr_th: float = 0.7
m_mask_corr_th: float = 0.5
explane_th: float = 0.5
m_pool_sample_start: int = 0  # internaly init as len(dates) - 70
m_pool_sample_end: int = -1

m_train_mode: int = 0  # [0 | 1] 0: init train, 1: Transfer
m_pre_train_model: str = "./save/model/rllearn/IF_Gra05_FOb3_MultiR_LR10E4_ICD_Gold_Buffer/20200205_1856_fs_epoch_4_0_pe1.53_pl1.43_vl1.15_ev0.775.pkl"
if m_train_mode == 1:
    m_name = m_name + "_continued"


"""Declare functions
"""
domain_search_parameter = {
    "INX_20": 1,
    "KS_20": 1,
    "Gold_20": 1,
    "FTSE_20": 1,
    "GDAXI_20": 1,
    "SSEC_20": 1,
    "BVSP_20": 1,
    "N225_20": 1,
    "INX_60": 2,
    "KS_60": 2,
    "Gold_60": 2,
    "FTSE_60": 2,
    "GDAXI_60": 2,
    "SSEC_60": 2,
    "BVSP_60": 2,
    "N225_60": 2,
    "INX_120": 3,
    "KS_120": 3,
    "Gold_120": 3,
    "FTSE_120": 3,
    "GDAXI_120": 3,
    "SSEC_120": 3,
    "BVSP_120": 3,
    "N225_120": 3,
    "US10YT_20": 4,
    "GB10YT_20": 4,
    "DE10YT_20": 4,
    "KR10YT_20": 4,
    "CN10YT_20": 4,
    "JP10YT_20": 4,
    "BR10YT_20": 4,
    "US10YT_60": 5,
    "GB10YT_60": 5,
    "DE10YT_60": 5,
    "KR10YT_60": 5,
    "CN10YT_60": 5,
    "JP10YT_60": 5,
    "BR10YT_60": 5,
    "US10YT_120": 6,
    "GB10YT_120": 6,
    "DE10YT_120": 6,
    "KR10YT_120": 6,
    "CN10YT_120": 6,
    "JP10YT_120": 6,
    "BR10YT_120": 6,
    "TOTAL_20": 7,
    "TOTAL_60": 8,
    "TOTAL_120": 9,
}
mkidx_mkname = {
    0: "INX",
    1: "KS",
    2: "Gold",
    3: "US10YT",
    4: "FTSE",
    5: "GDAXI",
    6: "SSEC",
    7: "BVSP",
    8: "N225",
    9: "GB10YT",
    10: "DE10YT",
    11: "KR10YT",
    12: "CN10YT",
    13: "JP10YT",
    14: "BR10YT",
    15: "TOTAL",
}
forward_map = {20: 1, 60: 2, 120: 3}
mkname_mkidx = {v: k for k, v in mkidx_mkname.items()}
mkname_dataset = {v: "v" + str(k + 11) for k, v in mkidx_mkname.items()}


def init_var(args) -> Tuple[int, str, str]:
    m_target_index: int = args.m_target_index
    target_name: str = target_id2name(m_target_index)
    m_name: str = objective + "_" + target_name
    return m_target_index, target_name, m_name


def target_id2name(m_target_index):
    return mkidx_mkname[m_target_index]


def get_file_name(m_target_index, file_data_vars) -> str:
    return file_data_vars + target_id2name(m_target_index) + "_intermediate.csv"


assert not (objective == "MT" and market_timing), "check environment setting"

if _debug_on:
    img_jpeg: Dict[str, int] = {
        "width": 1860,
        "height": 980,
        "dpi": 600,
    }  # full_jpeg = {'width': 1860, 'height': 980]
else:
    img_jpeg: Dict[str, int] = {
        "width": 1860,
        "height": 980,
        "dpi": 600,
    }  # full_jpeg = {'width': 1860, 'height': 980]
    # img_jpeg = {
    #     "width": 18,
    #     "height": 10,
    #     "dpi": 10,
    # }  # full_jpeg = {'width': 1860, 'height': 980]
