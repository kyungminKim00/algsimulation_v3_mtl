# -*- coding: utf-8 -*-
"""
@author: kim KyungMin
"""

from __future__ import absolute_import, division, print_function

import datetime
import os
import sys

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

# import matplotlib
# matplotlib.use('agg')
from matplotlib.dates import date2num

# from mpl_finance import candlestick_ohlc
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.metrics import classification_report, f1_score, mean_squared_error

from header.index_forecasting import RUNHEADER
from util import current_y_unit

# the data structure for res_info or tmp_info
data_dict = {
    "prediction_date": 0,
    "p_index": 1,
    "r_index": 2,
    "p_return": 3,
    "r_return": 4,
    "p_action": 5,
    "r_action": 6,
    "today_index": 7,
}


def gather_validation_performence(
    info,
    performence_info,
    values,
    # softmax_actions,
    # index_bound,
    # index_bound_return,
    # b_info,
    # b_info_return,
):
    r_action = info[0]["real_action"]
    p_action = info[0]["selected_action"]
    r_return = info[0]["20day_return"]
    p_return = values[0]
    r_index = info[0]["20day_index"]
    today_index = info[0]["today_index"]

    if current_y_unit(RUNHEADER.target_name) == "percent":
        p_index = p_return + today_index
    else:
        p_index = ((today_index * p_return) / 100) + today_index

    prediction_date = info[0]["p_date"]

    # data = [
    #         prediction_date,
    #         p_index,
    #         r_index,
    #         p_return,
    #         r_return,
    #         p_action[0],
    #         r_action[0],
    #         p_action[1],
    #         r_action[1],
    #         p_action[2],
    #         r_action[2],
    #         p_action[3],
    #         r_action[3],
    #         p_action[4],
    #         r_action[4],
    #         softmax_actions[0, 0, 0],
    #         softmax_actions[0, 0, 1],
    #         softmax_actions[0, 1, 0],
    #         softmax_actions[0, 1, 1],
    #         softmax_actions[0, 2, 0],
    #         softmax_actions[0, 2, 1],
    #         softmax_actions[0, 3, 0],
    #         softmax_actions[0, 3, 1],
    #         softmax_actions[0, 4, 0],
    #         softmax_actions[0, 4, 1],
    #         index_bound[0],
    #         index_bound[1],
    #         index_bound[2],
    #         index_bound[3],
    #         index_bound[4],
    #         index_bound_return[0],
    #         index_bound_return[1],
    #         index_bound_return[2],
    #         index_bound_return[3],
    #         index_bound_return[4],
    #         b_info,
    #         b_info_return,
    #         today_index,
    #     ]

    # np.array(r_action)[0,:].tolist()  # 20 up/down
    data = [
        prediction_date,
        p_index,
        r_index,
        p_return,
        r_return,
        p_action,
        np.array(r_action)[0, :].tolist(),
        # softmax_actions,
        # index_bound,
        # index_bound_return,
        # b_info,
        # b_info_return,
        today_index,
    ]
    performence_info.append(data)
    sys.stdout.write(f"\r>> (P) Test Date:  {prediction_date}")
    sys.stdout.flush()
    return performence_info


# def plot_prediction(
#     tmp_info,
#     save_dir,
#     current_model,
#     consistency,
#     correct_percent,
#     summary,
#     mse,
#     direction_f1,
#     ev,
#     direction_from_regression,
#     total,
# ):
#     # 1. index
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     # sub plot index
#     plt.subplot(2, 1, 1)
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(tmp_info[:, 0], tmp_info[:, 1].tolist(), label="prediction (index)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
#     # plt.plot(date, tmp_info[:, 38].tolist(), label='prediction2 (index)')
#     plt.legend()
#     # sub plot up/down
#     plt.subplot(2, 1, 2)
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(
#         tmp_info[:, 0], direction_from_regression, label="prediction (up/down)"
#     )  # from regression
#     plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
#     # plt.plot(tmp_info[:, 0], tmp_info[:, 5].tolist(), label='prediction (up/down)')
#     # plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label='real (up/down)')
#     plt.legend()

#     # plt.subplot(3, 1, 3)
#     # plt.xticks(np.arange(0, total, 40))
#     # plt.plot(tmp_info[:, 0], direction_prediction_label.tolist(), label='prediction (up/down)')
#     # plt.grid(True)
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_index/index/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#             ev,
#         ),
#         format="jpeg",
#         dpi=600,
#     )
#     plt.close()

#     # 2. returns
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     # # sub plot index 1
#     # plt.subplot(3, 1, 1)
#     # plt.xticks(np.arange(0, total, 40))
#     # plt.grid(True)
#     # plt.plot(date, tmp_info[:, 2].tolist(), label='real (index)')
#     # sub plot index 2
#     plt.subplot(2, 1, 1)
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(
#         tmp_info[:, 0], tmp_info[:, 3].tolist(), label="prediction (return)"
#     )  # from regression
#     plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (return)")
#     # plt.plot(date, tmp_info[:, 39].tolist(), label='prediction2 (return)')  # from regression
#     plt.legend()
#     # sub plot up/down
#     plt.subplot(2, 1, 2)
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(
#         tmp_info[:, 0], tmp_info[:, 5].tolist(), label="prediction (up/down)"
#     )  # from classifier
#     plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
#     plt.legend()
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_index/return/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#             ev,
#         ),
#         format="jpeg",
#         dpi=600,
#     )
#     plt.close()


def find_cur_band(
    date,
    data,
    predefined_std_return,
    predefined_std_index,
    mode="return",
    market_name=None,
):
    if mode == "return":
        predefined_std = predefined_std_return
    else:
        predefined_std = predefined_std_index

    predefined_std["TradeDate"] = pd.to_datetime(
        predefined_std["TradeDate"], format="%Y-%m-%d"
    )
    s_date = datetime.datetime.strftime(date[0], "%Y-%m-%d")
    e_date = datetime.datetime.strftime(date[-1], "%Y-%m-%d")

    varience = predefined_std.query("@s_date<= TradeDate <= @e_date")

    if str.upper(market_name) == "KS":
        market_name = "KS200"
    if str.upper(market_name) == "GOLD":
        market_name = "XAU"

    top = data + varience[market_name].values
    bottom = data - varience[market_name].values

    return top, bottom


def plot_prediction_band(
    tmp_info,
    prefix,
    current_model,
    consistency,
    correct_percent,
    summary,
    mse,
    direction_f1,
    ev,
    direction_from_regression,
    total,
    predefined_std_return,
    predefined_std_index,
    market_name,
):
    def top_bottom(data1, data2, date):
        assert (len(data1) == len(data2)) and (data1.shape == data2.shape)
        assert data1.shape[0] == len(date)

        date = date2num(np.array(date, dtype="datetime64"))

        if np.ndim(data1) == 1:
            open = np.expand_dims(data1, axis=0)
            close = np.expand_dims(data2, axis=0)
        else:
            raise IndexError("Check dimension")

        top = np.max(np.concatenate([open, close], axis=0), axis=0)
        bottom = np.min(np.concatenate([open, close], axis=0), axis=0)
        # tmp =[[date[idx], data1[idx], top[idx], bottom[idx], data2[idx]] for idx in range(len(date))]
        tmp = []
        # print('{:3.2}_{:3.2}_{:3.2}-{}-{}'.format(len(date), len(data1), len(data2), len(top), len(bottom)))
        for idx, item in enumerate(date):
            tmp.append([item, data1[idx], top[idx], bottom[idx], data2[idx]])
        return tmp

    date = [
        datetime.datetime.strptime(d_str, "%Y-%m-%d")
        for d_str in tmp_info[:, data_dict["prediction_date"]]
    ]

    # 1. index
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )

    # sub plot index
    top, bottom = find_cur_band(
        date,
        tmp_info[:, data_dict["p_index"]],
        predefined_std_return,
        predefined_std_index,
        mode="index",
        market_name=market_name,
    )
    ax1 = plt.subplot(2, 1, 1)
    # plt.xticks(np.arange(0, total, 40))  # Disable for mlp_finance
    plt.grid(True)
    plt.plot(date, tmp_info[:, data_dict["r_index"]].tolist(), label="real (index)")
    candlestick_ohlc(
        ax1,
        top_bottom(top, bottom, date),
        width=0.4,
        colorup="#77d879",
        colordown="#db3f3f",
    )
    plt.legend()
    # sub plot up/down
    plt.subplot(2, 1, 2)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(
        tmp_info[:, data_dict["prediction_date"]],
        direction_from_regression,
        label="prediction (up/down)",
    )  # from regression
    plt.plot(
        tmp_info[:, data_dict["prediction_date"]],
        tmp_info[:, data_dict["r_action"]].tolist(),
        label="real (up/down)",
    )
    plt.legend()
    plt.pause(1)

    f_name = f"{prefix}/index"
    if not os.path.isdir(f_name):
        os.makedirs(f_name)
    plt.savefig(
        f"{f_name}/{current_model}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}_{ev:3.2}.jpeg",
        format="jpeg",
        dpi=int(RUNHEADER.img_jpeg["dpi"]),
    )
    plt.close()

    # 2. returns
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    # sub plot index
    top, bottom = find_cur_band(
        date,
        tmp_info[:, data_dict["p_return"]],
        predefined_std_return,
        predefined_std_index,
        mode="return",
        market_name=market_name,
    )
    ax1 = plt.subplot(2, 1, 1)
    # plt.xticks(np.arange(0, total, 40))  # Disable for mlp_finance
    plt.grid(True)
    plt.plot(
        date,
        tmp_info[:, data_dict["r_return"]].tolist(),
        label=f"{current_model.split('_')[4]} real (return)",
    )
    candlestick_ohlc(
        ax1,
        top_bottom(top, bottom, date),
        width=0.4,
        colorup="#77d879",
        colordown="#db3f3f",
    )
    plt.legend()
    # sub plot up/down
    plt.subplot(2, 1, 2)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(
        tmp_info[:, data_dict["prediction_date"]],
        tmp_info[:, data_dict["p_action"]].tolist(),
        label="prediction (up/down)",
    )  # from classifier
    plt.plot(
        tmp_info[:, data_dict["prediction_date"]],
        tmp_info[:, data_dict["r_action"]].tolist(),
        label="real (up/down)",
    )
    plt.legend()
    plt.pause(1)

    f_name = f"{prefix}/return"
    if not os.path.isdir(f_name):
        os.makedirs(f_name)
    plt.savefig(
        f"{f_name}/{current_model}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}_{ev:3.2}.jpeg",
        format="jpeg",
        dpi=int(RUNHEADER.img_jpeg["dpi"]),
    )
    plt.close()


# # it may cause errors with correct incoming data, so modify code for plotting if go on
# def plot_bound_type1(
#     tmp_info,
#     save_dir,
#     current_model,
#     consistency,
#     correct_percent,
#     summary,
#     mse,
#     direction_f1,
#     total,
# ):
#     # 1. index
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(tmp_info[:, 0], tmp_info[:, 1].tolist(), label="prediction (index)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 25].tolist(), label="min (index)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 26].tolist(), label="max (index)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 27].tolist(), label="avg (index)")
#     plt.legend()
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_bound/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#         ),
#         format="jpeg",
#     )
#     plt.close()

#     # scatter plot
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     X = list()
#     Y = list()
#     for row in range(len(tmp_info)):
#         data = tmp_info[row, -2]
#         for col in range(len(data)):
#             X.append(tmp_info[row, 0])
#             Y.append(data[col])
#     plt.scatter(X, Y)
#     plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_scatter/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#         ),
#         format="jpeg",
#     )
#     plt.close()

#     # 2. return
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     plt.plot(tmp_info[:, 0], tmp_info[:, 3].tolist(), label="prediction (return)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (return)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 28].tolist(), label="min (return)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 29].tolist(), label="max (return)")
#     plt.plot(tmp_info[:, 0], tmp_info[:, 30].tolist(), label="avg (return)")
#     plt.legend()
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_bound/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#         ),
#         format="jpeg",
#     )
#     plt.close()

#     # scatter plot
#     plt_manager = plt.get_current_fig_manager()
#     plt_manager.resize(
#         int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
#     )
#     plt.xticks(np.arange(0, total, 40))
#     plt.grid(True)
#     X = list()
#     Y = list()
#     for row in range(len(tmp_info)):
#         data = tmp_info[row, -1]
#         for col in range(len(data)):
#             X.append(tmp_info[row, 0])
#             Y.append(data[col])
#     plt.scatter(X, Y)
#     plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (index)")
#     plt.pause(1)
#     plt.savefig(
#         "{}/fig_scatter/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
#             save_dir,
#             current_model,
#             consistency,
#             correct_percent,
#             summary,
#             mse,
#             direction_f1,
#         ),
#         format="jpeg",
#     )
#     plt.close()


def get_res_by_market(data, market_idx):
    res = [
        (
            it[data_dict["prediction_date"]],
            it[data_dict["p_index"]][market_idx],
            it[data_dict["r_index"]][market_idx],
            it[data_dict["p_return"]][market_idx],
            it[data_dict["r_return"]][market_idx],
            it[data_dict["p_action"]][market_idx],
            it[data_dict["r_action"]][market_idx],
            it[data_dict["today_index"]][market_idx],
        )
        for it in data
    ]

    return np.array(res, dtype=np.object)


@ray.remote
def ray_performence(
    tmp_info,
    save_dir,
    model_name,
    split_name,
    market_idx,
    predefined_std_index,
    predefined_std_return,
    current_model,
):
    """File out information"""
    # res_info = np.array(res_info, dtype=np.object)
    res_info = get_res_by_market(tmp_info, market_idx)
    market_name = RUNHEADER.mkidx_mkname[market_idx]

    df = pd.DataFrame(
        data=res_info,
        columns=[
            "P_Date",
            "P_index",
            "Index",
            "P_return",
            "Return",
            "P_20days",
            "20days",
            "today_index",
        ],
    )

    if split_name == "test":
        rtn = list(df["Return"])
        p_rtn = list(df["P_return"])
    else:
        rtn = list(df["Return"][-10:])
        p_rtn = list(df["P_return"][-10:])

    # up/down performance
    total = len(res_info)
    correct_percent = 1 - (
        np.sum(
            np.abs(
                np.array(res_info[:, 6].tolist()) - np.array(res_info[:, 5].tolist())
            )
        )
        / total
    )
    try:
        aa = res_info[:, 6].tolist()
        bb = res_info[:, 5].tolist()
        if np.allclose(aa, bb):
            if len(np.unique(aa)) == 1:
                aa = aa + [1]
                bb = bb + [1]
        summary_detail = classification_report(aa, bb, target_names=["Down", "Up"])
    except ValueError:
        summary_detail = None

    summary = f1_score(
        np.array(res_info[:, 6], dtype=np.int).tolist(),
        np.array(res_info[:, 5], dtype=np.int).tolist(),
        average="weighted",
    )

    consistency = 1 - (
        np.sum(
            np.abs(
                np.where(np.array(res_info[:, 3].tolist()) > 0, 1, 0)
                - np.array(res_info[:, 5].tolist())
            )
        )
        / total
    )

    # up/down from index forecasting
    direction_from_regression = np.where(res_info[:, 3] > 0, 1, 0).tolist()
    direction_real = np.diff(res_info[:, 4])
    direction_prediction = np.diff(res_info[:, 3])
    direction_real = np.append([direction_real[0]], direction_real)
    direction_prediction = np.append([np.zeros(1)], direction_prediction)
    direction_real_label = np.where(direction_real > 0, 1, 0)
    direction_prediction_label = np.where(direction_prediction > 0, 1, 0)
    direction_f1 = f1_score(
        np.array(direction_real_label, dtype=np.int).tolist(),
        np.array(direction_prediction_label, dtype=np.int).tolist(),
        average="weighted",
    )

    # returns error
    mse = mean_squared_error(res_info[:, 4].tolist(), res_info[:, 3].tolist())

    # ev
    ev = 1 - np.var(res_info[:, 4] - res_info[:, 3]) / np.var(res_info[:, 4])

    # ev for unseen validation only
    if split_name != "test":  # validation
        mse = mean_squared_error(res_info[-20:, 4].tolist(), res_info[-20:, 3].tolist())
        ev = 1 - np.var(res_info[-20:, 4] - res_info[-20:, 3]) / np.var(
            res_info[-20:, 4]
        )
        consistency = 1 - (
            np.sum(
                np.abs(
                    np.where(np.array(res_info[-20:, 3].tolist()) > 0, 1, 0)
                    - np.array(res_info[-20:, 5].tolist())
                )
            )
            / total
        )
        # regression up/down
        direction_real = np.diff(res_info[-20:, 4])
        direction_prediction = np.diff(res_info[-20:, 3])
        direction_real = np.append([direction_real[0]], direction_real)
        direction_prediction = np.append([np.zeros(1)], direction_prediction)
        direction_real_label = np.where(direction_real > 0, 1, 0)
        direction_prediction_label = np.where(direction_prediction > 0, 1, 0)
        direction_f1 = f1_score(
            np.array(direction_real_label, dtype=np.int).tolist(),
            np.array(direction_prediction_label, dtype=np.int).tolist(),
            average="weighted",
        )

    """File out
    """
    # save csv (total performance)
    prefix = f"{save_dir}/{market_name}"
    if not os.path.isdir(prefix):
        os.makedirs(prefix)
    prefix = f"{prefix}/{current_model}"

    df.to_csv(
        f"{prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}__{ev:3.2}.csv"
    )

    """Plot
    """
    if not RUNHEADER.m_bound_estimation:  # band with y prediction
        plot_prediction_band(
            res_info,
            prefix,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            ev,
            direction_from_regression,
            total,
            predefined_std_return,
            predefined_std_index,
            market_name,
        )

    # band with y prediction
    else:  # plot bound type 1 (this method takes tremendous time so not in use for now)
        assert False, "not defined yet, RUNHEADER.m_bound_estimation should be False"

    # text out for classification result
    with open(
        f"{prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}.txt",
        "w",
    ) as txt_fp:
        print(summary_detail, file=txt_fp)
        txt_fp.close()

    return {
        "consistency": consistency,
        "correct_percent": correct_percent,
        "summary": summary,
        "mse": mse,
        "ev": ev,
        "direction_f1": direction_f1,
        "rtn": rtn,
        "p_rtn": p_rtn,
    }


def plot_save_validation_performence(tmp_info, save_dir, model_name, split_name="test"):

    predefined_std_index = pd.read_csv(
        f"{RUNHEADER.predefined_std}{RUNHEADER.forward_ndx}_index.csv"
    )
    predefined_std_return = pd.read_csv(
        f"{RUNHEADER.predefined_std}{RUNHEADER.forward_ndx}_return.csv"
    )

    # current_model name
    current_model = model_name.split("/")[-1]
    current_model = current_model.split(".pkl")[0]
    current_model = current_model.split("_")
    current_model.remove(current_model[5])
    current_model = "_".join(current_model)

    res = [
        ray_performence.remote(
            tmp_info,
            save_dir,
            model_name,
            split_name,
            market_idx,
            predefined_std_index,
            predefined_std_return,
            current_model,
        )
        for market_idx in range(RUNHEADER.mtl_target)
    ]
    res = ray.get(res)

    total = {}
    for item_dict in res:
        for k, v in item_dict.items():
            try:
                total[k] = total[k] + [v]
            except KeyError:
                total[k] = [v]

    # file out total summary
    total_prefix = f"{save_dir}/{current_model}"
    consistency = np.mean(np.array(total["consistency"]))
    correct_percent = np.mean(np.array(total["correct_percent"]))
    f1 = np.mean(np.array(total["summary"]))
    mse = np.mean(np.array(total["mse"]))
    ev = np.mean(np.array(total["ev"]))
    direction_f1 = np.mean(np.array(total["direction_f1"]))
    total_rtn = np.array(total["rtn"]).reshape(-1)
    total_p_rtn = np.array(total["p_rtn"]).reshape(-1)

    pd.DataFrame(
        data=np.array([total_rtn, total_p_rtn]).T, columns=["Return", "P_return"]
    ).to_csv(
        f"{total_prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{f1:3.2}___{mse:3.2}_{direction_f1:3.2}__{ev:3.2}.csv"
    )


def Original_RAY_적용전_plot_save_validation_performence(
    tmp_info, save_dir, model_name, split_name="test"
):
    # toal performence
    (
        total_consistency,
        total_correct_percent,
        total_f1,
        total_mse,
        total_ev,
        total_direction_f1,
        total_rtn,
        total_p_rtn,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        (),
        (),
    )

    predefined_std_index = pd.read_csv(
        f"{RUNHEADER.predefined_std}{RUNHEADER.forward_ndx}_index.csv"
    )
    predefined_std_return = pd.read_csv(
        f"{RUNHEADER.predefined_std}{RUNHEADER.forward_ndx}_return.csv"
    )

    for market_idx in range(RUNHEADER.mtl_target):
        """File out information"""
        # res_info = np.array(res_info, dtype=np.object)
        res_info = get_res_by_market(tmp_info, market_idx)
        market_name = RUNHEADER.mkidx_mkname[market_idx]

        df = pd.DataFrame(
            data=res_info,
            columns=[
                "P_Date",
                "P_index",
                "Index",
                "P_return",
                "Return",
                "P_20days",
                "20days",
                # "P_10days",
                # "10days",
                # "P_15days",
                # "15days",
                # "P_25days",
                # "25days",
                # "P_30days",
                # "30days",
                # "P_Confidence",
                # "P_Probability",
                # "P_10days_False_Confidence",
                # "10days_True_Confidence",
                # "P_15days_False_Confidence",
                # "15days_True_Confidence",
                # "P_25days_False_Confidence",
                # "25days_True_Confidence",
                # "P_30days_False_Confidence",
                # "30days_True_Confidence",
                # "min",
                # "max",
                # "avg",
                # "std",
                # "median",
                # "min_return",
                # "max_return",
                # "avg_return",
                # "std_return",
                # "median_return",
                # "b_info",
                # "b_info_return",
                "today_index",
            ],
        )

        if split_name == "test":
            total_rtn = total_rtn + tuple(df["Return"])
            total_p_rtn = total_p_rtn + tuple(df["P_return"])
        else:
            total_rtn = total_rtn + tuple(df["Return"][-10:])
            total_p_rtn = total_p_rtn + tuple(df["P_return"][-10:])

        # up/down performance
        total = len(res_info)
        half = int(total * 0.5)
        correct_percent = 1 - (
            np.sum(
                np.abs(
                    np.array(res_info[:, 6].tolist())
                    - np.array(res_info[:, 5].tolist())
                )
            )
            / total
        )
        try:
            aa = res_info[:, 6].tolist()
            bb = res_info[:, 5].tolist()
            if np.allclose(aa, bb):
                if len(np.unique(aa)) == 1:
                    aa = aa + [1]
                    bb = bb + [1]
            summary_detail = classification_report(aa, bb, target_names=["Down", "Up"])
        except ValueError:
            summary_detail = None
            pass

        summary = f1_score(
            np.array(res_info[:, 6], dtype=np.int).tolist(),
            np.array(res_info[:, 5], dtype=np.int).tolist(),
            average="weighted",
        )
        cf_val = f1_score(
            np.array(res_info[:half, 6], dtype=np.int).tolist(),
            np.array(res_info[:half, 5], dtype=np.int).tolist(),
            average="weighted",
        )
        cf_test = f1_score(
            np.array(res_info[half:, 6], dtype=np.int).tolist(),
            np.array(res_info[half:, 5], dtype=np.int).tolist(),
            average="weighted",
        )
        consistency = 1 - (
            np.sum(
                np.abs(
                    np.where(np.array(res_info[:, 3].tolist()) > 0, 1, 0)
                    - np.array(res_info[:, 5].tolist())
                )
            )
            / total
        )

        # up/down from index forecasting
        direction_from_regression = np.where(res_info[:, 3] > 0, 1, 0).tolist()
        direction_real = np.diff(res_info[:, 4])
        direction_prediction = np.diff(res_info[:, 3])
        direction_real = np.append([direction_real[0]], direction_real)
        direction_prediction = np.append([np.zeros(1)], direction_prediction)
        direction_real_label = np.where(direction_real > 0, 1, 0)
        direction_prediction_label = np.where(direction_prediction > 0, 1, 0)
        direction_f1 = f1_score(
            np.array(direction_real_label, dtype=np.int).tolist(),
            np.array(direction_prediction_label, dtype=np.int).tolist(),
            average="weighted",
        )

        # returns error
        mse = mean_squared_error(res_info[:, 4].tolist(), res_info[:, 3].tolist())
        mse_val = mean_squared_error(
            res_info[:half, 4].tolist(), res_info[:half, 3].tolist()
        )
        mse_test = mean_squared_error(
            res_info[half:, 4].tolist(), res_info[half:, 3].tolist()
        )

        # subtract T/F confidences
        # subtract_confidences_p = np.array(tmp_info[:, 16].tolist()) - np.array(tmp_info[:, 15].tolist())
        # min_max_subtract_confidences = (subtract_confidences_p - np.min(subtract_confidences_p)) / \
        #                                (np.max(subtract_confidences_p) - np.min(subtract_confidences_p) + 1E-4)
        # differ_subtract_confidences = np.diff(min_max_subtract_confidences)
        # differ_subtract_confidences = np.insert(differ_subtract_confidences, 0, 0)
        # differ_subtract_confidences_normal_p = (differ_subtract_confidences -
        #                                         np.mean(differ_subtract_confidences) + 1E-15) / \
        #                                        np.std(differ_subtract_confidences)
        # differ_subtract_condifences_tan_p = np.tan(differ_subtract_confidences_normal_p)

        subtract_confidences_p = 0
        min_max_subtract_confidences = 0
        differ_subtract_confidences = 0
        differ_subtract_confidences = 0
        differ_subtract_confidences_normal_p = 0
        differ_subtract_condifences_tan_p = 0

        # ev
        ev = 1 - np.var(res_info[:, 4] - res_info[:, 3]) / np.var(res_info[:, 4])
        ev_val = 1 - np.var(res_info[:half, 4] - res_info[:half, 3]) / np.var(
            res_info[:half, 4]
        )
        ev_test = 1 - np.var(res_info[half:, 4] - res_info[half:, 3]) / np.var(
            res_info[half:, 4]
        )

        # ev for unseen validation only
        if split_name != "test":  # validation
            mse = mean_squared_error(
                res_info[-20:, 4].tolist(), res_info[-20:, 3].tolist()
            )
            ev = 1 - np.var(res_info[-20:, 4] - res_info[-20:, 3]) / np.var(
                res_info[-20:, 4]
            )
            consistency = 1 - (
                np.sum(
                    np.abs(
                        np.where(np.array(res_info[-20:, 3].tolist()) > 0, 1, 0)
                        - np.array(res_info[-20:, 5].tolist())
                    )
                )
                / total
            )
            # regression up/down
            direction_real = np.diff(res_info[-20:, 4])
            direction_prediction = np.diff(res_info[-20:, 3])
            direction_real = np.append([direction_real[0]], direction_real)
            direction_prediction = np.append([np.zeros(1)], direction_prediction)
            direction_real_label = np.where(direction_real > 0, 1, 0)
            direction_prediction_label = np.where(direction_prediction > 0, 1, 0)
            direction_f1 = f1_score(
                np.array(direction_real_label, dtype=np.int).tolist(),
                np.array(direction_prediction_label, dtype=np.int).tolist(),
                average="weighted",
            )

        """File out
        """
        # save csv (total performance)
        current_model = model_name.split("/")[-1]
        current_model = current_model.split(".pkl")[0]
        current_model = current_model.split("_")
        current_model.remove(current_model[5])
        current_model = "_".join(current_model)
        prefix = f"{save_dir}/{market_name}"
        if not os.path.isdir(prefix):
            os.makedirs(prefix)
        prefix = f"{prefix}/{current_model}"

        df.to_csv(
            f"{prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}__{ev:3.2}.csv"
        )

        total_consistency.append(consistency)
        total_correct_percent.append(correct_percent)
        total_f1.append(summary)
        total_mse.append(mse)
        total_ev.append(ev)
        total_direction_f1.append(direction_f1)

        # pname = "{}_CF[{:3.2}_{:3.2}_{:3.2}]_RE[{:3.2}_{:3.2}_{:3.2}]_EV[{:3.2}_{:3.2}_{:3.2}].txt".format(
        #     prefix,
        #     cf_val,
        #     cf_test,
        #     summary,
        #     mse_val,
        #     mse_test,
        #     mse,
        #     ev_val,
        #     ev_test,
        #     ev,
        # )
        # fp = open(pname, "w")
        # print("validation-test-total performance", file=fp)
        # fp.close()

        """Plot
        """
        # plot_file_prefix = current_model.replace('_fs_epoch', '_ep')
        # file_name_list = plot_file_prefix.split('_')
        # file_name_list.remove(file_name_list[5])
        # file_name_list.remove(file_name_list[4])
        # plot_file_prefix = '_'.join(file_name_list)
        # plot_file_prefix = current_model
        if not RUNHEADER.m_bound_estimation:  # band with y prediction
            if RUNHEADER.m_bound_estimation_y:  # band estimation (plot bound type 2)
                plot_prediction_band(
                    res_info,
                    prefix,
                    current_model,
                    consistency,
                    correct_percent,
                    summary,
                    mse,
                    direction_f1,
                    ev,
                    direction_from_regression,
                    total,
                    predefined_std_return,
                    predefined_std_index,
                    market_name,
                )
            else:  # point estimation
                assert (
                    False
                ), "not defined yet, RUNHEADER.m_bound_estimation_y should be True"
                # plot_prediction(
                #     res_info,
                #     save_dir,
                #     plot_file_prefix,
                #     consistency,
                #     correct_percent,
                #     summary,
                #     mse,
                #     direction_f1,
                #     ev,
                #     direction_from_regression,
                #     total,
                # )
        # band with y prediction
        else:  # plot bound type 1 (this method takes tremendous time so not in use for now)
            assert (
                False
            ), "not defined yet, RUNHEADER.m_bound_estimation should be False"
            # plot_bound_type1(
            #     res_info,
            #     save_dir,
            #     plot_file_prefix,
            #     consistency,
            #     correct_percent,
            #     summary,
            #     mse,
            #     direction_f1,
            #     total,
            # )

        # text out for classification result
        with open(
            f"{prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{summary:3.2}___{mse:3.2}_{direction_f1:3.2}.txt",
            "w",
        ) as txt_fp:
            print(summary_detail, file=txt_fp)
            txt_fp.close()

    # file out total summary
    total_prefix = f"{save_dir}/{current_model}"
    consistency = np.mean(np.array(total_consistency))
    correct_percent = np.mean(np.array(total_correct_percent))
    f1 = np.mean(np.array(total_f1))
    mse = np.mean(np.array(total_mse))
    ev = np.mean(np.array(total_ev))
    direction_f1 = np.mean(np.array(total_direction_f1))

    pd.DataFrame(
        data=np.array([total_rtn, total_p_rtn]).T, columns=["Return", "P_return"]
    ).to_csv(
        f"{total_prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{f1:3.2}___{mse:3.2}_{direction_f1:3.2}__{ev:3.2}.csv"
    )
    # with open(
    #     f"{total_prefix}_C_{consistency:3.2}___{correct_percent:3.2}_{f1:3.2}___{mse:3.2}_{direction_f1:3.2}__{ev:3.2}.csv",
    #     "w",
    # ) as txt_fp:
    #     print("", file=txt_fp)
    #     txt_fp.close()


# def epoch_summary(save_dir):
#     def find_col_idx(cols, col_name):
#         return [idx for idx in range(len(cols)) if cols[idx] == col_name][0]

#     list_dict = {
#         "epoch": 4,
#         "val_EV": 18,
#         "PL": 5,
#         "VL": 6,
#         "EV": 7,
#         "consistency": 9,
#         "CF": 13,
#         "RE": 16,
#         "RF": 17,
#     }
#     data_rows_column = list(list_dict.keys())
#     file_location = "{}/return/".format(save_dir)

#     # text tokenizing
#     data_rows = []
#     for performence in os.listdir(file_location):
#         if "jpeg" in performence:
#             torken = str.split(performence, "_")
#             assert (
#                 len(torken) == 19
#             ), "file name format may be changed.. check it: {}".format(torken)

#             tmp = list()
#             for list_idx in list(list_dict.values()):
#                 if (
#                     list_dict["EV"] == list_idx
#                     or list_dict["VL"] == list_idx
#                     or list_dict["PL"] == list_idx
#                 ):
#                     p_data = float(torken[list_idx][2:])
#                 else:
#                     if "jpeg" in torken[list_idx]:
#                         p_data = float(torken[list_idx][:-5])
#                     else:
#                         p_data = float(torken[list_idx])
#                 tmp.append(p_data)
#             data_rows.append(tmp)
#     assert not len(data_rows) == 0, FileNotFoundError("check file names")

#     # summary performance by epoch
#     data_rows = np.array(data_rows)
#     colname_epoch_idx = find_col_idx(data_rows_column, "epoch")
#     epoch_idx = sorted(list(set(data_rows[:, colname_epoch_idx].tolist())))
#     epoch_performance = list()
#     for epoch in epoch_idx:
#         row_idxs = (
#             np.argwhere(data_rows[:, colname_epoch_idx] == epoch).squeeze().tolist()
#         )
#         tmp = data_rows[row_idxs]
#         if np.ndim(tmp) == 1:
#             epoch_performance.append(tmp.tolist())
#         else:
#             epoch_performance.append(np.mean(tmp, axis=0).tolist())

#     # file out
#     # align data
#     file_out_column_name = [
#         "epoch",
#         "val_EV",
#         "PL",
#         "VL",
#         "EV",
#         "consistency",
#         "CF",
#         "RE",
#         "RF",
#     ]
#     file_out_columns_idx = [
#         find_col_idx(data_rows_column, s_name) for s_name in file_out_column_name
#     ]
#     epoch_performance = np.array(epoch_performance).T[file_out_columns_idx].T.round(3)

#     # file out
#     pd.DataFrame(data=epoch_performance, columns=file_out_column_name).to_csv(
#         file_location + "summary_by_epoch.csv"
#     )


# def naive_filter(model_list):
#     filtered_model = list()
#     for model_name in model_list:
#         token = model_name.split('_')
#         try:
#             if float(token[6][2:]) < 3 and float(token[7][2:]) < 3 and \
#                     float(token[8][2:]) < 1.6 and float(token[9][2:-4]) > 0.9:
#                 filtered_model.append(model_name)
#         except IndexError:
#             filtered_model.append(model_name)
#     [filtered_model.remove(model) for model in filtered_model if '.pkl' not in model]
#
#     return sorted(list(set(filtered_model)), key=len)

# def adhoc_process(model_files, result_files):
#     model_files = os.listdir('./save/model/rllearn/IF_TGold_FVar_20200319_1121_SParm_Buffer_20200322_0119')
#     filenames = list()
#     [filenames.append(_model) for _model in model_files if '.pkl' and 'sub_epo' in _model]
#     filenames.sort()
#
#     result_files = './save/result/IF_TGold_FVar_20200314_1232_FParm_Buffer_20200314_2315/fig_index/return/'
#     # gather models
#     # assume proper models are selected already
#     selected_model_lists = list()
#     predicted_index, predicted_keys = list(), list()
#     model_lists = naive_filter(model_files)
#     for result_file in result_files:
#         for model_list in model_lists:
#             if model_list in result_file:
#                 selected_model_lists.append(model_list)
#
#     # accumulate performances
#     for selected_model in selected_model_lists:
#         predicted_keys = pd.read_csv(selected_model).keys()
#         predicted_index_tmp = pd.read_csv(selected_model).values[:, 1:].tolist()
#         predicted_index.append(predicted_index_tmp)
#     predicted_index = np.array(predicted_index)
#     predicted_index_summary = np.mean(predicted_index, axis=2)
#     out_file = result_files.split('/')[:-2] + '/adhoc'
#     pd.DataFrame(data=predicted_index_summary, columns=predicted_keys).to_csv(out_file)


# if __name__ == '__main__':
#     epoch_summary('./save/result/IF_TGold_FVar_20200319_1121_SParm_Buffer_20200322_0119')
#     adhoc_process('model_files', 'result_files')
