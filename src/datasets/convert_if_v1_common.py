from __future__ import absolute_import, division, print_function

import copy
import math
import pickle
import warnings
from collections import OrderedDict

import bottleneck as bn
import numpy as np
import pandas as pd
import ray
import statsmodels.api as sm
from header.index_forecasting import RUNHEADER
from util import (
    current_y_unit,
    get_conjunction_dates_data_v3,
    get_working_dates,
    ordinary_return,
)

from datasets.unit_datetype_des_check import write_var_desc
from datasets.windowing import fun_cross_cov, rolling_apply_cross_cov

_NUM_SHARDS = 5


class ReadData(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(
        self,
        date,
        data,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    ):
        self.source_data = data
        self.target_data = target_data
        self.x_seq = x_seq
        self.class_names_to_ids = class_names_to_ids
        self.date = date
        self.forward_ndx = forward_ndx
        self.ref_forward_ndx = ref_forward_ndx

    def _get_returns(self, p_data, n_data, unit="prc"):
        return ordinary_return(v_init=p_data, v_final=n_data, unit=unit)

    def _get_class_seq(self, data, base_date, interval, unit="prc"):
        tmp = []
        for days in interval:
            tmp.append(
                self._get_returns(
                    data[base_date, :],
                    data[base_date + self.forward_ndx + days, :],
                    unit=unit,
                )
            )
        return np.array(tmp, dtype=np.float32)

    def _get_normal(self, data):
        std = np.std(np.array(data, dtype=np.float), axis=0)
        std = np.where(std == 0, 1e-12, std)
        normal_data = (data - np.mean(data, axis=0) + 1e-12) / std
        assert np.allclose(data.shape, normal_data.shape)
        return normal_data

    # Crop Data
    def _get_patch(self, base_date, train_sample=True, historical_y=False):

        x_start_ndx = base_date - self.x_seq + 1
        x_end_ndx = base_date + 1

        """X data Section
        """
        # given source data
        if self.source_data.ndim == 1:
            self.data = self.source_data[x_start_ndx:x_end_ndx]
            _ = self.data.shape
        elif self.source_data.ndim == 2:
            self.data = self.source_data[
                x_start_ndx:x_end_ndx, :
            ]  # x_seq+1 by the num of variables
            self.height, self.width = self.data.shape
        elif self.source_data.ndim == 3:
            self.data = self.source_data[x_start_ndx:x_end_ndx, :, :]
            _, self.height, self.width = self.data.shape
        else:
            assert False, "None defined dimension!!!"

        if historical_y:
            self.normal_data = self._get_normal(self.data)
            # daily return
            previous_data = (
                self.source_data[x_start_ndx - 1 : x_end_ndx - 1]
                if self.source_data.ndim == 1
                else self.source_data[x_start_ndx - 1 : x_end_ndx - 1, :]
            )
            self.data_diff = ordinary_return(
                v_init=previous_data,
                v_final=self.data,
                unit=current_y_unit(RUNHEADER.target_name),
            )
            self.normal_data_diff = self._get_normal(self.data_diff)
            self.data_mu = np.mean(self.data_diff)
            self.data_sigma = np.var(self.data_diff)
            # reshape - extract today statistics
            (
                self.normal_data,
                self.data_diff,
                self.normal_data_diff,
                self.data_mu,
                self.data_sigma,
            ) = (
                np.expand_dims(self.normal_data[-1], axis=0),
                np.expand_dims(self.data_diff[-1], axis=0),
                np.expand_dims(self.normal_data_diff[-1], axis=0),
                np.expand_dims(self.data_mu, axis=0),
                np.expand_dims(self.data_sigma, axis=0),
            )
        else:
            # Apply standardization
            self.normal_data = self._get_normal(self.data)

            # Apply status_data for 5 days (returns)
            previous_data = (
                self.source_data[x_start_ndx - 5 : x_end_ndx - 5]
                if self.source_data.ndim == 1
                else self.source_data[x_start_ndx - 5 : x_end_ndx - 5, :]
            )
            self.status_data5 = ordinary_return(v_init=previous_data, v_final=self.data)

            # patch min, max
            self.patch_min = np.min(self.data, axis=0)
            self.patch_max = np.max(self.data, axis=0)

        """Y data Section
        """
        unit = current_y_unit(RUNHEADER.target_name)

        # y_seq by the num of variables
        self.class_seq_price = self.target_data[
            base_date + 1 : base_date + self.forward_ndx + 1, :
        ]
        self.class_seq_height, self.class_seq_width = self.class_seq_price.shape

        backward_ndx = 5
        self.tr_class_seq_price_minmaxNor = self.target_data[
            base_date - backward_ndx : base_date + self.forward_ndx + 1, :
        ]
        self.tr_class_seq_price_minmaxNor = (
            self.tr_class_seq_price_minmaxNor[backward_ndx, :]
            - self.tr_class_seq_price_minmaxNor.min(axis=0)
        ) / (
            self.tr_class_seq_price_minmaxNor.max(axis=0)
            - self.tr_class_seq_price_minmaxNor.min(axis=0)
        )

        self.class_index = self.target_data[
            base_date + self.forward_ndx, :
        ]  # +20 days Price(index)
        self.tr_class_index = self.tr_class_seq_price_minmaxNor
        self.base_date_price = self.target_data[base_date, :]  # +0 days Price(index)

        self.class_ratio = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + self.forward_ndx, :],
            unit=unit,
        )  # +20 days
        self.class_ratio_ref3 = self._get_returns(
            self.target_data[base_date - 1, :],
            self.target_data[base_date, :],
            unit=unit,
        )  # today
        self.class_ratio_ref1 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + self.forward_ndx + self.ref_forward_ndx[0], :],
            unit=unit,
        )  # +10 days
        self.class_ratio_ref2 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + self.forward_ndx + self.ref_forward_ndx[1], :],
            unit=unit,
        )  # +15 days
        self.class_label = np.where(
            self.class_ratio > 0, 1, 0
        )  # +20 days up/down label
        self.class_label_ref1 = np.where(
            self.class_ratio_ref1 > 0, 1, 0
        )  # +10 days up/down label
        self.class_label_ref2 = np.where(
            self.class_ratio_ref2 > 0, 1, 0
        )  # +15 days up/down label
        self.class_label_ref3 = np.where(
            self.class_ratio_ref3 > 0, 1, 0
        )  # +0 days up/down label

        # self.tr_class_label_call = np.where(
        #     self.tr_class_index <= 0.2, 1, 0
        # )  # call label
        # self.tr_class_label_hold = np.where(
        #     (self.tr_class_index > 0.2) & (self.tr_class_index < 0.8), 1, 0
        # )  # hold label
        # self.tr_class_label_put = np.where(
        #     self.tr_class_index >= 0.8, 1, 0
        # )  # put label

        if train_sample:
            self.class_seq_ratio = self._get_class_seq(
                self.target_data, base_date, [-2, -1, 0, 1, 2], unit=unit
            )
            self.class_ratio_ref4 = self._get_returns(
                self.target_data[base_date, :],
                self.target_data[
                    base_date + self.forward_ndx + self.ref_forward_ndx[2], :
                ],
                unit=unit,
            )  # +25 days
            self.class_ratio_ref5 = self._get_returns(
                self.target_data[base_date, :],
                self.target_data[
                    base_date + self.forward_ndx + self.ref_forward_ndx[3], :
                ],
                unit=unit,
            )  # +30 days
            self.class_label_ref4 = np.where(
                self.class_ratio_ref4 > 0, 1, 0
            )  # +25 days up/down label
            self.class_label_ref5 = np.where(
                self.class_ratio_ref5 > 0, 1, 0
            )  # +30 days up/down label
        else:
            self.class_seq_ratio = self._get_class_seq(
                self.target_data, base_date, [-2, -1, 0], unit=unit
            )
            self.class_ratio_ref4 = None
            self.class_ratio_ref5 = None
            self.class_label_ref4 = None  # +25 days up/down label
            self.class_label_ref5 = None  # +30 days up/down label

        """Date data Section
        """
        self.base_date_index = base_date
        self.base_date_label = self.date[base_date]
        self.prediction_date_index = base_date + self.forward_ndx
        self.prediction_date_label = self.date[base_date + self.forward_ndx]

    def get_patch(self, base_date, train_sample=True, historical_y=False):
        # initialize variables
        self.data = None
        self.height = None
        self.width = None
        self.normal_data = None
        self.status_data5 = None
        self.status_data5_Y = None
        self.diff_data = None
        self.patch_min = None
        self.patch_max = None
        self.normal_data_diff = None
        self.data_diff = None
        self.data_mu = None
        self.data_sigma = None

        self.class_seq_height = None
        self.class_seq_width = None
        self.class_seq_price = None
        self.tr_class_seq_price_minmaxNor = None
        self.class_seq_ratio = None

        self.class_index = None
        self.tr_class_index = None
        self.class_ratio = None
        self.class_ratio_ref0 = None
        self.class_ratio_ref1 = None
        self.class_ratio_ref2 = None
        self.class_ratio_ref3 = None
        self.class_ratio_ref4 = None
        self.class_ratio_ref5 = None
        self.class_ratio_ref6 = None
        self.class_ratio_ref7 = None

        self.class_label = None
        self.class_label_ref1 = None
        self.class_label_ref2 = None
        self.class_label_ref3 = None
        self.class_label_ref4 = None
        self.class_label_ref5 = None
        self.tr_class_label_call = None
        self.tr_class_label_hold = None
        self.tr_class_label_put = None
        self.class_name = None

        self.base_date_price = None
        self.base_date_label = None
        self.base_date_index = None
        self.prediction_date_label = None
        self.prediction_date_index = None

        # extract a patch
        self._get_patch(base_date, train_sample, historical_y)


def cut_off_data(
    data,
    cut_off,
    blind_set_seq=None,
    test_s_date=None,
    test_e_date=None,
    operation_mode=False,
    forward_ndx=None,
    ref_forward_ndx=None,
):
    eof = len(data)
    dummy_date = (
        forward_ndx if forward_ndx > ref_forward_ndx[-1] else ref_forward_ndx[-1]
    )
    dummy_date = dummy_date + 5  # for his return of x (feature)

    if operation_mode:
        data = add_data_4_operation(data, test_e_date)

    if test_s_date is None:
        blind_set_seq = eof - blind_set_seq
        if len(data.shape) == 1:  # 1D
            tmp = (
                data[cut_off:blind_set_seq],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :
                ],
            )
        elif len(data.shape) == 2:  # 2D:
            tmp = (
                data[cut_off:blind_set_seq, :],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :,
                    :,
                ],
            )
        elif len(data.shape) == 3:  # 3D:
            tmp = (
                data[cut_off:blind_set_seq, ::],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :,
                    :,
                    :,
                ],
            )
        else:
            raise IndexError("Define your cut-off code")
    else:
        if test_e_date is None or test_s_date == test_e_date:
            if len(data.shape) == 1:  # 1D
                tmp = (
                    data[cut_off:test_s_date],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :
                    ],
                )
            elif len(data.shape) == 2:  # 2D:
                tmp = (
                    data[cut_off:test_s_date, :],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :,
                        :,
                    ],
                )
            elif len(data.shape) == 3:  # 3D:
                tmp = (
                    data[cut_off:test_s_date, ::],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :,
                        :,
                        :,
                    ],
                )
            else:
                raise IndexError("Define your cut-off code")
        else:  # s_date, e_date are given
            if len(data.shape) == 1:  # 1D
                tmp = (
                    data[cut_off:test_s_date],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date
                    ],
                )
            elif len(data.shape) == 2:  # 2D:
                tmp = (
                    data[cut_off:test_s_date, :],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date,
                        :,
                    ],
                )
            elif len(data.shape) == 3:  # 3D:
                tmp = (
                    data[cut_off:test_s_date, ::],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date,
                        :,
                        :,
                    ],
                )
            else:
                raise IndexError("Define your cut-off code")
    return tmp


def add_data_4_operation(data, test_e_date=None):
    n_length = test_e_date - data.shape[0]

    if data.ndim == 1:
        add_data = np.zeros([n_length])
    elif data.ndim == 2:
        add_data = np.zeros([n_length, data.shape[1]])
    elif data.ndim == 3:
        add_data = np.zeros([n_length, data.shape[1], data.shape[2]])
    else:
        assert False, "check dimensions"
    return np.append(data, add_data, axis=0)


def cv_index_configuration(date, verbose, forward_ndx, ref_forward_ndx):
    num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
    start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
    if verbose == 0:  # train and validation separately
        for shard_id in range(_NUM_SHARDS):
            start_end_index_list[shard_id] = [
                shard_id * num_per_shard,
                min((shard_id + 1) * num_per_shard, len(date)),
            ]
    elif verbose == 1:  # test
        start_end_index_list[0] = [0, len(date)]
    elif verbose == 2:  # from 0 to end - only train without validation
        start_end_index_list[0] = [0, len(date)]
    elif verbose == 3:  # duplicated validation for early stopping criteria
        headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
        duplicated_samples = (
            -40 - headbias_from_y_excluded - RUNHEADER.m_warm_up_4_inference
        )
        start_end_index_list[0] = [0, len(date)]
        start_end_index_list[1] = [len(date) + duplicated_samples, len(date)]
    elif verbose == 4:  # train and validation separately
        headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
        val_samples = -250 - headbias_from_y_excluded  # val samples 1years
        start_end_index_list[0] = [0, len(date) - val_samples]
        start_end_index_list[1] = [len(date) + val_samples, len(date)]
    return _cv_index_configuration(start_end_index_list, verbose), verbose


def _cv_index_configuration(start_end_index_list, verbose):
    index_container = []
    validation = []
    train = []
    if verbose == 0:  # train and validation
        for idx, _ in enumerate(start_end_index_list):
            for ckeck_idx, val in enumerate(start_end_index_list):
                if ckeck_idx == idx:
                    validation.append(val)
                else:
                    train.append(val)
            index_container.append([validation, train])
            validation = list()
            train = list()
    elif verbose == 1:
        index_container = start_end_index_list
    elif verbose == 2:
        index_container = start_end_index_list
    elif (
        verbose == 3 or verbose == 4
    ):  # index_container contains validation and training
        index_container.append([start_end_index_list[1], start_end_index_list[0]])
    return index_container


def load_file(file_location, file_format):
    with open(file_location, "rb") as fp:
        if file_format == "npy":
            data = np.load(fp)
            fp.close()
            return data
        elif file_format == "pkl":
            data = pickle.load(fp)
            fp.close()
            return data
        else:
            raise ValueError("non-support file format")


@ray.remote
def ray_wrap_fun(fun, ma_data, x_idx, num_cov_obs):
    return rolling_apply_cross_cov(fun, ma_data[:, x_idx], ma_data[:, -1], num_cov_obs)[
        :, 0, 1
    ]


@ray.remote
def ray_wrap_lr(X, Y):
    warnings.filterwarnings("ignore")
    result = sm.OLS(Y, X).fit()
    return [np.abs(result.params[0]), result.rsquared]


def _getcorr(
    data,
    target_data,
    base_first_momentum,
    num_cov_obs,
    b_scaler=True,
    opt_mask=None,
    y_idx=None,
):
    _data = np.hstack([data, np.expand_dims(target_data, axis=1)])
    ma_data = bn.move_mean(
        _data, window=base_first_momentum, min_count=1, axis=0
    )  # use whole train samples

    # the variable selection with cross corealation
    print(
        f"[{ma_data.shape[1] - 1} vars with {RUNHEADER.target_id2name(y_idx)}] cross corrleation"
    )

    res = [
        ray_wrap_fun.remote(fun_cross_cov, ma_data, x_idx, num_cov_obs)
        for x_idx in range(ma_data.shape[1] - 1)
    ]
    new_cov = np.array(ray.get(res)).T

    mean_cor_pivot = bn.move_mean(
        new_cov, window=RUNHEADER.m_pool_samples * 2, min_count=1, axis=0
    )
    std_cor_pivot = bn.move_std(
        new_cov, window=RUNHEADER.m_pool_samples * 2, min_count=1, axis=0
    )

    # determin marginal area
    upper = mean_cor_pivot + std_cor_pivot * RUNHEADER.var_select_factor
    lower = mean_cor_pivot - std_cor_pivot * RUNHEADER.var_select_factor

    mean_cor = bn.move_mean(
        new_cov, window=RUNHEADER.m_pool_samples, min_count=1, axis=0
    )

    daily_cov_raw = mean_cor
    tmp_cov = np.where((mean_cor > upper) | (mean_cor < lower), 1, 0)
    tmp_cov[: RUNHEADER.m_pool_samples * 2, :] = 0

    print(f"the avg. numbers of employed variables: {np.mean(np.sum(tmp_cov, axis=1))}")

    return tmp_cov, daily_cov_raw


def get_corr(data, target_data, x_unit=None, b_scaler=True, opt_mask=None):
    base_first_momentum, num_cov_obs = 5, 20  # default
    mask = []

    num_y_var = target_data.shape[1]
    for y_idx in range(num_y_var):
        tmp_cov, daily_cov_raw = _getcorr(
            data,
            target_data[:, y_idx],
            base_first_momentum,
            num_cov_obs,
            b_scaler,
            opt_mask,
            y_idx,
        )

        # if x_unit is not None:
        #     add_vol_index = np.array(x_unit) == "volatility"
        #     tmp_cov = add_vol_index + tmp_cov
        #     tmp_cov = np.where(tmp_cov >= 1, 1, 0)

        mask.append(tmp_cov.tolist())
        # mean_cov = np.nanmean(tmp_cov, axis=0)
        # cov_dict = dict(zip(list(ids_to_var_names.values()), mean_cov.tolist()))
        # cov_dict = OrderedDict(sorted(cov_dict.items(), key=lambda x: x[1], reverse=True))
    mask = np.transpose(np.array(mask), [1, 2, 0])
    assert not np.any(mask > 1), "validate mask error!!, it can not be bigger than 1"

    daily_num_var = np.sum(mask, axis=1)
    mean_num = np.mean(daily_num_var, axis=0, dtype=int)
    std_num = np.std(daily_num_var, axis=0, dtype=float)
    print(f"the mean of variables on daily for 15 markets: {mean_num}")
    print(f"the std of variables on daily for 15 markets: {std_num}")
    return mask, daily_cov_raw


def ma(data):
    # windowing for sd_data, according to the price
    ma_data_5 = bn.move_mean(data, window=5, min_count=1, axis=0)
    ma_data_10 = bn.move_mean(data, window=10, min_count=1, axis=0)
    ma_data_20 = bn.move_mean(data, window=20, min_count=1, axis=0)
    ma_data_60 = bn.move_mean(data, window=60, min_count=1, axis=0)

    return ma_data_5, ma_data_10, ma_data_20, ma_data_60


def normalized_spread(data, ma_data_5, ma_data_10, data_20, ma_data_60, X_unit):
    f1, f2, f3, f4, f5 = (
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
    )
    ma_data_3 = bn.move_mean(
        data, window=3, min_count=1, axis=0
    )  # 3days moving average

    for idx, _unit in enumerate(X_unit):
        f1[:, idx] = ordinary_return(
            v_init=ma_data_3[:, idx], v_final=data[:, idx], unit=_unit
        )
        f2[:, idx] = ordinary_return(
            v_init=ma_data_5[:, idx], v_final=data[:, idx], unit=_unit
        )
        f3[:, idx] = ordinary_return(
            v_init=ma_data_10[:, idx], v_final=data[:, idx], unit=_unit
        )
        f4[:, idx] = ordinary_return(
            v_init=data_20[:, idx], v_final=data[:, idx], unit=_unit
        )
        f5[:, idx] = ordinary_return(
            v_init=ma_data_60[:, idx], v_final=data[:, idx], unit=_unit
        )

    return f1, f2, f3, f4, f5


def _get_index_df(v, index_price, ids_to_var_names, target_data=None):
    x1, x2 = None, None
    is_exist = False

    for idx, _ in enumerate(ids_to_var_names):
        if v == ids_to_var_names[idx]:
            return index_price[:, idx]

    if not is_exist:
        # assert is_exist, "could not find a given variable name: {}".format(v)
        return np.zeros(index_price.shape[0])


def get_index_df(index_price=None, ids_to_var_names=None, c_name=None):
    visit_once = 0
    for market in list(RUNHEADER.mkidx_mkname.values()):
        if market in c_name:
            target_name = market
            visit_once = visit_once + 1
            print(f"gather variables for {target_name}")

            assert visit_once == 1, "not_allow_duplication"

    c_name = pd.read_csv(c_name, header=None)
    c_name = c_name.values.squeeze().tolist()

    if RUNHEADER.re_assign_vars:
        new_vars = []

        if RUNHEADER.manual_vars_additional:
            manual = pd.read_csv(f"{RUNHEADER.file_data_vars}MANUAL_Indices.csv")
            manual_vars = list(manual.values.reshape(-1))
            new_vars = c_name + manual_vars
            new_vars = list(dict.fromkeys(new_vars))

        c_name = OrderedDict(
            sorted(zip(new_vars, range(len(new_vars))), key=lambda aa: aa[1])
        )

        # save var list
        file_name = RUNHEADER.file_data_vars + target_name
        pd.DataFrame(data=list(c_name.keys()), columns=["VarName"]).to_csv(
            file_name + "_Indices_v1.csv",
            index=None,
            header=None,
        )
        print(f"{file_name}_Indices_v1.csv has been saved")

        # save var desc
        d_f_summary = pd.read_csv(RUNHEADER.var_desc)
        basename = (file_name + "_Indices_v1.csv").split(".csv")[0]
        write_var_desc(list(c_name.keys()), d_f_summary, basename)
    else:
        c_name = OrderedDict(
            sorted(zip(c_name, range(len(c_name))), key=lambda aa: aa[1])
        )

    index_df = [_get_index_df(v, index_price, ids_to_var_names) for v in c_name.keys()]
    index_df = np.array(index_df, dtype=np.float32).T

    # check not in source
    nis = np.sum(index_df, axis=0) == 0
    c_nis = np.where(nis == True, False, True)
    index_df = index_df[:, c_nis]
    c_name = np.array(list(c_name.keys()))[c_nis].tolist()

    return np.array(index_df, dtype=np.float32), OrderedDict(
        sorted(zip(range(len(c_name)), c_name), key=lambda aa: aa[0])
    )


def splite_rawdata_v1(index_price=None, y_index=None, c_name=None):
    index_df = pd.read_csv(index_price)
    index_df = index_df.ffill(axis=0)
    index_df = index_df.bfill(axis=0)
    index_dates = index_df.values[:, 0]
    index_values = np.array(index_df.values[:, 1:], dtype=np.float32)
    ids_to_var_names = OrderedDict(
        zip(range(len(index_df.keys()[1:])), index_df.keys()[1:])
    )

    y_index_df = pd.read_csv(y_index)
    y_index_df = y_index_df.ffill(axis=0)
    y_index_df = y_index_df.bfill(axis=0)
    y_index_dates = y_index_df.values[:, 0]
    y_index_values = np.array(y_index_df.values[:, 1:], dtype=np.float32)
    ids_to_class_names = OrderedDict(
        zip(range(len(y_index_df.keys()[1:])), y_index_df.keys()[1:])
    )

    # get working dates
    index_dates, index_values = get_working_dates(index_dates, index_values)
    y_index_dates, y_index_values = get_working_dates(y_index_dates, y_index_values)

    # the conjunction of target and independent variables
    dates, sd_data, y_index_dates, y_index_data = get_conjunction_dates_data_v3(
        index_dates, y_index_dates, index_values, y_index_values
    )
    # according to the data type of dependent variables, generate return values
    num_y_var = y_index_data.shape[1]
    returns = np.zeros(y_index_data.shape)
    for y_idx in range(num_y_var):
        target_name = RUNHEADER.target_id2name(y_idx)
        unit = current_y_unit(target_name)
        rtn = ordinary_return(matrix=y_index_data, unit=unit)  # daily return
        returns[:, y_idx] = rtn[:, y_idx]

    rtn_tuple = (None, None, None, None, None, None)
    if c_name is not None:
        for c_name_var in c_name:
            _sd_data, _ids_to_var_names = get_index_df(
                sd_data, ids_to_var_names, c_name_var
            )

            if "data_vars_TOTAL_Indices.csv" in c_name_var:
                rtn_tuple = (
                    dates,
                    copy.deepcopy(_sd_data),
                    y_index_data,
                    returns,
                    ids_to_class_names,
                    copy.deepcopy(_ids_to_var_names),
                )

    return rtn_tuple
