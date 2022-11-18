# -*- coding: utf-8 -*-
"""
@author: kim KyungMin
"""

from __future__ import absolute_import, division, print_function

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('agg')
from matplotlib.dates import date2num

# from mpl_finance import candlestick_ohlc
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.metrics import classification_report, f1_score, mean_squared_error

import header.index_forecasting.RUNHEADER as RUNHEADER
from util import current_y_unit


def gather_validation_performence(
    info,
    tmp_info,
    values,
    values2,
    softmax_actions,
    index_bound,
    index_bound_return,
    b_info,
    b_info_return,
):
    r_action = info[0]["real_action"]
    p_action = info[0]["selected_action"]
    r_return = info[0]["20day_return"]
    p_return = values[0]
    p_return2 = values2[0]
    r_index = info[0]["20day_index"]
    today_index = info[0]["today_index"]

    if current_y_unit(RUNHEADER.target_name) == "percent":
        p_index = p_return + today_index
        p_index2 = p_return2 + today_index

    else:
        p_index = ((today_index * p_return) / 100) + today_index
        p_index2 = ((today_index * p_return2) / 100) + today_index

    prediction_date = info[0]["p_date"]

    tmp_info.append(
        [
            prediction_date,
            p_index,
            r_index,
            p_return,
            r_return,
            p_action[0],
            r_action[0],
            p_action[1],
            r_action[1],
            p_action[2],
            r_action[2],
            p_action[3],
            r_action[3],
            p_action[4],
            r_action[4],
            softmax_actions[0, 0, 0],
            softmax_actions[0, 0, 1],
            softmax_actions[0, 1, 0],
            softmax_actions[0, 1, 1],
            softmax_actions[0, 2, 0],
            softmax_actions[0, 2, 1],
            softmax_actions[0, 3, 0],
            softmax_actions[0, 3, 1],
            softmax_actions[0, 4, 0],
            softmax_actions[0, 4, 1],
            index_bound[0],
            index_bound[1],
            index_bound[2],
            index_bound[3],
            index_bound[4],
            index_bound_return[0],
            index_bound_return[1],
            index_bound_return[2],
            index_bound_return[3],
            index_bound_return[4],
            b_info,
            b_info_return,
            today_index,
            p_index2,
            p_return2,
        ]
    )
    sys.stdout.write("\r>> (P) Test Date:  %s" % prediction_date)
    sys.stdout.flush()
    return tmp_info


def plot_prediction(
    tmp_info,
    save_dir,
    current_model,
    consistency,
    correct_percent,
    summary,
    mse,
    direction_f1,
    ev,
    direction_from_regression,
    total,
):
    # 1. index
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    # sub plot index
    plt.subplot(2, 1, 1)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(tmp_info[:, 0], tmp_info[:, 1].tolist(), label="prediction (index)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
    # plt.plot(date, tmp_info[:, 38].tolist(), label='prediction2 (index)')
    plt.legend()
    # sub plot up/down
    plt.subplot(2, 1, 2)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(
        tmp_info[:, 0], direction_from_regression, label="prediction (up/down)"
    )  # from regression
    plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
    # plt.plot(tmp_info[:, 0], tmp_info[:, 5].tolist(), label='prediction (up/down)')
    # plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label='real (up/down)')
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.xticks(np.arange(0, total, 40))
    # plt.plot(tmp_info[:, 0], direction_prediction_label.tolist(), label='prediction (up/down)')
    # plt.grid(True)
    plt.pause(1)
    plt.savefig(
        "{}/fig_index/index/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            ev,
        ),
        format="jpeg",
        dpi=600,
    )
    plt.close()

    # 2. returns
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    # # sub plot index 1
    # plt.subplot(3, 1, 1)
    # plt.xticks(np.arange(0, total, 40))
    # plt.grid(True)
    # plt.plot(date, tmp_info[:, 2].tolist(), label='real (index)')
    # sub plot index 2
    plt.subplot(2, 1, 1)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(
        tmp_info[:, 0], tmp_info[:, 3].tolist(), label="prediction (return)"
    )  # from regression
    plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (return)")
    # plt.plot(date, tmp_info[:, 39].tolist(), label='prediction2 (return)')  # from regression
    plt.legend()
    # sub plot up/down
    plt.subplot(2, 1, 2)
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(
        tmp_info[:, 0], tmp_info[:, 5].tolist(), label="prediction (up/down)"
    )  # from classifier
    plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
    plt.legend()
    plt.pause(1)
    plt.savefig(
        "{}/fig_index/return/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            ev,
        ),
        format="jpeg",
        dpi=600,
    )
    plt.close()


def plot_prediction_band(
    tmp_info,
    save_dir,
    current_model,
    consistency,
    correct_percent,
    summary,
    mse,
    direction_f1,
    ev,
    direction_from_regression,
    total,
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
        tmp = list()
        # print('{:3.2}_{:3.2}_{:3.2}-{}-{}'.format(len(date), len(data1), len(data2), len(top), len(bottom)))
        for idx in range(len(date)):
            tmp.append([date[idx], data1[idx], top[idx], bottom[idx], data2[idx]])
        return tmp

    date = [datetime.datetime.strptime(d_str, "%Y-%m-%d") for d_str in tmp_info[:, 0]]

    # 1. index
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    # sub plot index
    ax1 = plt.subplot(2, 1, 1)
    # plt.xticks(np.arange(0, total, 40))  # Disable for mlp_finance
    plt.grid(True)
    plt.plot(date, tmp_info[:, 2].tolist(), label="real (index)")
    candlestick_ohlc(
        ax1,
        top_bottom(tmp_info[:, 1], tmp_info[:, 38], date),
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
        tmp_info[:, 0], direction_from_regression, label="prediction (up/down)"
    )  # from regression
    plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
    plt.legend()

    plt.pause(1)
    plt.savefig(
        "{}/fig_index/index/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            ev,
        ),
        format="jpeg",
        dpi=int(RUNHEADER.img_jpeg["dpi"]),
    )
    plt.close()

    # 2. returns
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    ax1 = plt.subplot(2, 1, 1)
    # plt.xticks(np.arange(0, total, 40))  # Disable for mlp_finance
    plt.grid(True)
    plt.plot(
        date,
        tmp_info[:, 4].tolist(),
        label="{} real (return)".format(current_model.split("_")[4]),
    )
    candlestick_ohlc(
        ax1,
        top_bottom(tmp_info[:, 3], tmp_info[:, 39], date),
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
        tmp_info[:, 0], tmp_info[:, 5].tolist(), label="prediction (up/down)"
    )  # from classifier
    plt.plot(tmp_info[:, 0], tmp_info[:, 6].tolist(), label="real (up/down)")
    plt.legend()
    plt.pause(1)
    plt.savefig(
        "{}/fig_index/return/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            ev,
        ),
        format="jpeg",
        dpi=int(RUNHEADER.img_jpeg["dpi"]),
    )
    plt.close()


# it may cause errors but incoming data is correct so modify code for plotting if go on
def plot_bound_type1(
    tmp_info,
    save_dir,
    current_model,
    consistency,
    correct_percent,
    summary,
    mse,
    direction_f1,
    total,
):
    # 1. index
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(tmp_info[:, 0], tmp_info[:, 1].tolist(), label="prediction (index)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 25].tolist(), label="min (index)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 26].tolist(), label="max (index)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 27].tolist(), label="avg (index)")
    plt.legend()
    plt.pause(1)
    plt.savefig(
        "{}/fig_bound/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
        ),
        format="jpeg",
    )
    plt.close()

    # scatter plot
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    X = list()
    Y = list()
    for row in range(len(tmp_info)):
        data = tmp_info[row, -2]
        for col in range(len(data)):
            X.append(tmp_info[row, 0])
            Y.append(data[col])
    plt.scatter(X, Y)
    plt.plot(tmp_info[:, 0], tmp_info[:, 2].tolist(), label="real (index)")
    plt.pause(1)
    plt.savefig(
        "{}/fig_scatter/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
        ),
        format="jpeg",
    )
    plt.close()

    # 2. return
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    plt.plot(tmp_info[:, 0], tmp_info[:, 3].tolist(), label="prediction (return)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (return)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 28].tolist(), label="min (return)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 29].tolist(), label="max (return)")
    plt.plot(tmp_info[:, 0], tmp_info[:, 30].tolist(), label="avg (return)")
    plt.legend()
    plt.pause(1)
    plt.savefig(
        "{}/fig_bound/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
        ),
        format="jpeg",
    )
    plt.close()

    # scatter plot
    plt_manager = plt.get_current_fig_manager()
    plt_manager.resize(
        int(RUNHEADER.img_jpeg["width"]), int(RUNHEADER.img_jpeg["height"])
    )
    plt.xticks(np.arange(0, total, 40))
    plt.grid(True)
    X = list()
    Y = list()
    for row in range(len(tmp_info)):
        data = tmp_info[row, -1]
        for col in range(len(data)):
            X.append(tmp_info[row, 0])
            Y.append(data[col])
    plt.scatter(X, Y)
    plt.plot(tmp_info[:, 0], tmp_info[:, 4].tolist(), label="real (index)")
    plt.pause(1)
    plt.savefig(
        "{}/fig_scatter/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.jpeg".format(
            save_dir,
            current_model,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
        ),
        format="jpeg",
    )
    plt.close()


def plot_save_validation_performence(tmp_info, save_dir, model_name, split_name="test"):
    """File out information"""
    tmp_info = np.array(tmp_info, dtype=np.object)
    df = pd.DataFrame(
        data=tmp_info,
        columns=[
            "P_Date",
            "P_index",
            "Index",
            "P_return",
            "Return",
            "P_20days",
            "20days",
            "P_10days",
            "10days",
            "P_15days",
            "15days",
            "P_25days",
            "25days",
            "P_30days",
            "30days",
            "P_Confidence",
            "P_Probability",
            "P_10days_False_Confidence",
            "10days_True_Confidence",
            "P_15days_False_Confidence",
            "15days_True_Confidence",
            "P_25days_False_Confidence",
            "25days_True_Confidence",
            "P_30days_False_Confidence",
            "30days_True_Confidence",
            "min",
            "max",
            "avg",
            "std",
            "median",
            "min_return",
            "max_return",
            "avg_return",
            "std_return",
            "median_return",
            "b_info",
            "b_info_return",
            "today_index",
            "P_index2",
            "P_return2",
        ],
    )
    # up/down performance
    total = len(tmp_info)
    half = int(total * 0.5)
    correct_percent = 1 - (
        np.sum(
            np.abs(
                np.array(tmp_info[:, 6].tolist()) - np.array(tmp_info[:, 5].tolist())
            )
        )
        / total
    )
    try:
        aa = tmp_info[:, 6].tolist()
        bb = tmp_info[:, 5].tolist()
        if np.allclose(aa, bb):
            if len(np.unique(aa)) == 1:
                aa = aa + [1]
                bb = bb + [1]
        summary_detail = classification_report(aa, bb, target_names=["Down", "Up"])
    except ValueError:
        summary_detail = None
        pass

    summary = f1_score(
        np.array(tmp_info[:, 6], dtype=np.int).tolist(),
        np.array(tmp_info[:, 5], dtype=np.int).tolist(),
        average="weighted",
    )
    cf_val = f1_score(
        np.array(tmp_info[:half, 6], dtype=np.int).tolist(),
        np.array(tmp_info[:half, 5], dtype=np.int).tolist(),
        average="weighted",
    )
    cf_test = f1_score(
        np.array(tmp_info[half:, 6], dtype=np.int).tolist(),
        np.array(tmp_info[half:, 5], dtype=np.int).tolist(),
        average="weighted",
    )
    consistency = 1 - (
        np.sum(
            np.abs(
                np.where(np.array(tmp_info[:, 3].tolist()) > 0, 1, 0)
                - np.array(tmp_info[:, 5].tolist())
            )
        )
        / total
    )

    # up/down from index forecasting
    direction_from_regression = np.where(tmp_info[:, 3] > 0, 1, 0).tolist()
    direction_real = np.diff(tmp_info[:, 4])
    direction_prediction = np.diff(tmp_info[:, 3])
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
    mse = mean_squared_error(tmp_info[:, 4].tolist(), tmp_info[:, 3].tolist())
    mse_val = mean_squared_error(
        tmp_info[:half, 4].tolist(), tmp_info[:half, 3].tolist()
    )
    mse_test = mean_squared_error(
        tmp_info[half:, 4].tolist(), tmp_info[half:, 3].tolist()
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
    ev = 1 - np.var(tmp_info[:, 4] - tmp_info[:, 3]) / np.var(tmp_info[:, 4])
    ev_val = 1 - np.var(tmp_info[:half, 4] - tmp_info[:half, 3]) / np.var(
        tmp_info[:half, 4]
    )
    ev_test = 1 - np.var(tmp_info[half:, 4] - tmp_info[half:, 3]) / np.var(
        tmp_info[half:, 4]
    )

    # ev for unseen validation only
    if split_name != "test":  # validation
        mse = mean_squared_error(tmp_info[-20:, 4].tolist(), tmp_info[-20:, 3].tolist())
        ev = 1 - np.var(tmp_info[-20:, 4] - tmp_info[-20:, 3]) / np.var(
            tmp_info[-20:, 4]
        )
        consistency = 1 - (
            np.sum(
                np.abs(
                    np.where(np.array(tmp_info[-20:, 3].tolist()) > 0, 1, 0)
                    - np.array(tmp_info[-20:, 5].tolist())
                )
            )
            / total
        )
        # regression up/down
        direction_real = np.diff(tmp_info[-20:, 4])
        direction_prediction = np.diff(tmp_info[-20:, 3])
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
    prefix = save_dir + "/" + current_model
    df.to_csv(
        prefix
        + "_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}___{:3.2}.csv".format(
            consistency, correct_percent, summary, mse, ev
        )
    )
    pname = "{}_CF[{:3.2}_{:3.2}_{:3.2}]_RE[{:3.2}_{:3.2}_{:3.2}]_EV[{:3.2}_{:3.2}_{:3.2}].txt".format(
        prefix, cf_val, cf_test, summary, mse_val, mse_test, mse, ev_val, ev_test, ev
    )
    fp = open(pname, "w")
    print("validation-test-total performance", file=fp)
    fp.close()

    """Plot
    """
    # plot_file_prefix = current_model.replace('_fs_epoch', '_ep')
    # file_name_list = plot_file_prefix.split('_')
    # file_name_list.remove(file_name_list[5])
    # file_name_list.remove(file_name_list[4])
    # plot_file_prefix = '_'.join(file_name_list)
    plot_file_prefix = current_model
    if not RUNHEADER.m_bound_estimation:  # band with y prediction
        if RUNHEADER.m_bound_estimation_y:  # band estimation (plot bound type 2)
            plot_prediction_band(
                tmp_info,
                save_dir,
                plot_file_prefix,
                consistency,
                correct_percent,
                summary,
                mse,
                direction_f1,
                ev,
                direction_from_regression,
                total,
            )
        else:  # point estimation
            plot_prediction(
                tmp_info,
                save_dir,
                plot_file_prefix,
                consistency,
                correct_percent,
                summary,
                mse,
                direction_f1,
                ev,
                direction_from_regression,
                total,
            )
    # band with y prediction
    else:  # plot bound type 1 (this method takes tremendous time so not in use for now)
        plot_bound_type1(
            tmp_info,
            save_dir,
            plot_file_prefix,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
            total,
        )

    # text out for classification result
    with open(
        "{}/{}_C_{:3.2}___{:3.2}_{:3.2}___{:3.2}_{:3.2}.txt".format(
            save_dir,
            plot_file_prefix,
            consistency,
            correct_percent,
            summary,
            mse,
            direction_f1,
        ),
        "w",
    ) as txt_fp:
        print(summary_detail, file=txt_fp)
        txt_fp.close()


def epoch_summary(save_dir):
    def find_col_idx(cols, col_name):
        return [idx for idx in range(len(cols)) if cols[idx] == col_name][0]

    list_dict = {
        "epoch": 4,
        "val_EV": 18,
        "PL": 5,
        "VL": 6,
        "EV": 7,
        "consistency": 9,
        "CF": 13,
        "RE": 16,
        "RF": 17,
    }
    data_rows_column = list(list_dict.keys())
    file_location = "{}/fig_index/return/".format(save_dir)

    # text tokenizing
    data_rows = list()
    for performence in os.listdir(file_location):
        if "jpeg" in performence:
            torken = str.split(performence, "_")
            assert (
                len(torken) == 19
            ), "file name format may be changed.. check it: {}".format(torken)

            tmp = list()
            for list_idx in list(list_dict.values()):
                if (
                    list_dict["EV"] == list_idx
                    or list_dict["VL"] == list_idx
                    or list_dict["PL"] == list_idx
                ):
                    p_data = float(torken[list_idx][2:])
                else:
                    if "jpeg" in torken[list_idx]:
                        p_data = float(torken[list_idx][:-5])
                    else:
                        p_data = float(torken[list_idx])
                tmp.append(p_data)
            data_rows.append(tmp)
    assert not len(data_rows) == 0, FileNotFoundError("check file names")

    # summary performance by epoch
    data_rows = np.array(data_rows)
    colname_epoch_idx = find_col_idx(data_rows_column, "epoch")
    epoch_idx = sorted(list(set(data_rows[:, colname_epoch_idx].tolist())))
    epoch_performance = list()
    for epoch in epoch_idx:
        row_idxs = (
            np.argwhere(data_rows[:, colname_epoch_idx] == epoch).squeeze().tolist()
        )
        tmp = data_rows[row_idxs]
        if np.ndim(tmp) == 1:
            epoch_performance.append(tmp.tolist())
        else:
            epoch_performance.append(np.mean(tmp, axis=0).tolist())

    # file out
    # align data
    file_out_column_name = [
        "epoch",
        "val_EV",
        "PL",
        "VL",
        "EV",
        "consistency",
        "CF",
        "RE",
        "RF",
    ]
    file_out_columns_idx = [
        find_col_idx(data_rows_column, s_name) for s_name in file_out_column_name
    ]
    epoch_performance = np.array(epoch_performance).T[file_out_columns_idx].T.round(3)

    # file out
    pd.DataFrame(data=epoch_performance, columns=file_out_column_name).to_csv(
        file_location + "summary_by_epoch.csv"
    )


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
