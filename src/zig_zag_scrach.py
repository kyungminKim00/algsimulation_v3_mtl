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
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Bypass the need to install Tkinter GUI framework

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import print_flush
import argparse


def animate_gif(source_dir, duration=100):
    width = 640 * 0.5
    height = 480 * 0.5

    img, *imgs = None, None
    t_dir = source_dir

    # filepaths
    fp_in = t_dir + "/*.png"
    fp_out = t_dir + "/animated.gif"

    try:
        print("Reading:{}".format(fp_in))
        img, *imgs = [
            Image.open(f).resize((int(width), int(height)))
            for f in sorted(glob.glob(fp_in))
        ]
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=duration,
        )
        img.close()
    except ValueError:
        print("ValueError:{}".format(fp_in))
        pass


def t_r_c_d(cnt, f_bool=True, expected_label=None):
    if (len(realized_returns) > 0) and (cnt >= start_idx) and (cnt <= end_idx):
        if f_bool:
            tmp_expected_return = expected_returns[-1]
        else:
            tmp_expected_return = 0
        total_return_chart_data.append(
            [realized_returns[-1], tmp_expected_return, TradeDate[cnt]]
        )
        historical_return.append(
            [realized_returns[-1], tmp_expected_return, expected_label, TradeDate[cnt]]
        )


def total_return_chart_plot(plt):
    tmp_data = np.array(total_return_chart_data)
    realized_y = np.array(tmp_data[:, 0], dtype=np.float).tolist()
    realized_y = np.cumsum(realized_y)
    expected_y = np.array(tmp_data[:, 1], dtype=np.float).tolist()
    expected_y = np.cumsum(expected_y)
    dates = np.array(tmp_data[:, 2]).tolist()

    (fig, ax) = plt.subplots()
    ax.plot(dates, realized_y, color="black", label="realized_y")
    plt.xticks(np.arange(0, len(dates), 1000))

    ax.plot(
        dates,
        expected_y,
        color="red",
        label="expected_y",
    )

    # Save graph to file.
    plt.title(
        "Accuracy: {}, (BM) Return:{}, (Exptect) Return:{}".format(
            cul_performance[-1],
            round(np.sum(realized_y[-1]), 3),
            round(np.sum(expected_y[-1]), 3),
        )
    )
    plt.legend(loc="best")

    img_name = "total_return_mv{}_T{}.png".format(mv, forward)
    plt.savefig("./{}/{}".format(store_dir, img_name))
    plt.close()


def plot(plt, expected_label, label, returns, wrong_sample):
    if expected_label == label:
        performance_returns = np.abs(returns)
    else:
        performance_returns = -np.abs(returns)
    expected_returns.append(performance_returns)
    expected_returns_std.append([performance_returns, returns, std])

    def _wrong_sample(wrong_sample, performance_returns):
        global wrong_sample_cnt
        if wrong_sample:
            if performance_returns < 0:
                wrong_sample_cnt = wrong_sample_cnt + 1
                _plot(plt, expected_label)
        else:
            _plot(plt, expected_label)

    # Plot condition
    if save_section_eable:
        if save_section:
            _wrong_sample(wrong_sample, performance_returns)
        else:
            pass
    else:
        _wrong_sample(wrong_sample, performance_returns)


def _plot(plt, expected_label):
    # Plot main graph.
    (fig, ax) = plt.subplots()
    # ax.plot(data_x, data_y)
    ax.plot(data_x_s, data_y_s)
    plt.xticks(np.arange(0, len(data_x_s), 60))

    # Plot peaks.
    peak_x = peak_indexes
    peak_y = data_y[peak_indexes]
    ax.plot(
        peak_x,
        peak_y,
        marker="o",
        linestyle="dashed",
        color="green",
        label="Peaks",
    )

    # Plot valleys.
    valley_x = valley_indexes
    valley_y = data_y[valley_indexes]
    ax.plot(
        valley_x,
        valley_y,
        marker="o",
        linestyle="dashed",
        color="blue",
        label="Valleys",
    )

    # mark
    ax.plot(
        mark_x_s,
        mark_y_s,
        marker="o",
        linestyle="dashed",
        color="black",
        label="T",
    )

    if expected_label:
        color = "red"
    else:
        color = "blue"
    ax.plot(
        mark_x_s_prediction,
        mark_y_s_prediction,
        marker="o",
        linestyle="dashed",
        color=color,
        label="T+{}".format(str(forward)),
    )

    # Save graph to file.
    # plt.title(
    #     "Accuracy: {}, (BM) Return:{}, (Exptect) Return:{}, std:{}".format(
    #         cul_performance[-1],
    #         round(np.sum(realized_returns), 3),
    #         round(np.sum(expected_returns), 3),
    #         std
    #     )
    # )
    plt.title(
        "Stength: {}, (BM) Return:{}, (Exptect) Return:{}, std:{}".format(
            round(stength, 3),
            round(np.sum(realized_returns), 3),
            round(np.sum(expected_returns), 3),
            std
        )
    )
    plt.legend(loc="best")

    img_name = "open_close_mv{}_T{}_{}_{}_{}.png".format(mv, forward, str(cnt), TradeDate[cnt], std)
    print_flush("Generating: {} {} ".format(img_name, TradeDate[cnt]))
    plt.savefig("./{}/{}".format(store_dir, img_name))
    plt.close()


def rule_simulation(
    peak_indexes,
    valley_indexes,
    raw_vals,
    cnt,
    forward,
    bins,
    plt,
    wrong_sample=False,
    t_e=0,
    enable_check_sum=False,
    method=None,
):
    label = np.where(raw_vals[cnt + forward] - raw_vals[cnt] < 0, 0, 1)
    returns = ((raw_vals[cnt + forward] - raw_vals[cnt]) / raw_vals[cnt]) * 100
    realized_returns.append(returns)

    def rule_simulation_R1(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std

        def get_method(check_sum, expected_label, _method=None):
            if not enable_check_sum:
                check_sum = expected_label
            if _method == "negative":
                return (check_sum == expected_label) and (expected_label == 0)
            elif _method == "positive":
                return (check_sum == expected_label) and (expected_label == 1)
            else:  # bots
                return check_sum == expected_label

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0:
            t_r_c_d(cnt, False, 2)
        else:
            std = data_y[peak_indexes].std(ddof=1)
            expected_label = np.where(peak_indexes[-1] > valley_indexes[-1], 0, 1)
            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)

            if get_method(check_sum, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)

    def rule_simulation_R2(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std

        def get_method(check_sum, check_sum_2, expected_label, _method=None):
            if not enable_check_sum:
                if enable_check_sum:
                    check_sum = expected_label
                if _method == "negative":
                    return (check_sum == expected_label) and (expected_label == 0)
                elif _method == "positive":
                    return (check_sum == expected_label) and (expected_label == 1)
                else:  # bots
                    return check_sum == expected_label
            return False

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0:
            t_r_c_d(cnt, False, 2)
        else:
            std = data_y[peak_indexes].std(ddof=1)
            expected_label = np.where(peak_indexes[-1] > valley_indexes[-1], 0, 1)
            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)
            if (
                raw_vals[cnt] > data_y[peak_indexes[-1]]
                or raw_vals[cnt] < data_y[valley_indexes[-1]]
            ):
                check_sum_2 = True
            else:
                check_sum_2 = False

            if get_method(check_sum, check_sum_2, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)

    def rule_simulation_R3(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std

        def get_method(check_sum, expected_label, _method=None):
            if not enable_check_sum:
                check_sum = expected_label
            if _method == "negative":
                return (check_sum == expected_label) and (expected_label == 0)
            elif _method == "positive":
                return (check_sum == expected_label) and (expected_label == 1)
            else:  # bots
                return check_sum == expected_label

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0:
            t_r_c_d(cnt, False, 2)
        else:
            std = data_y[peak_indexes].std(ddof=1)
            expected_label = np.where(peak_indexes[-1] > valley_indexes[-1], 0, 1)
            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)

            if get_method(check_sum, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)

    def rule_simulation_R4(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        bins,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std

        def get_method(check_sum, expected_label, _method=None):
            if not enable_check_sum:
                check_sum = expected_label
            if _method == "negative":
                return (check_sum == expected_label) and (expected_label == 0)
            elif _method == "positive":
                return (check_sum == expected_label) and (expected_label == 1)
            else:  # bots
                return check_sum == expected_label

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0 or len(valley_indexes) < 7:
            t_r_c_d(cnt, False, 2)
        else:
            tmp = np.concatenate((data_y[peak_indexes], data_y[valley_indexes]))
            std = tmp.std(ddof=1)
            short_bin, long_bin = 3, 13

            peak_path = np.append(data_y[peak_indexes], raw_vals[cnt])
            valley_path = np.append(data_y[valley_indexes], raw_vals[cnt])
            if np.mean(valley_path[-short_bin:]) > np.mean(valley_path[-long_bin:]):
                expected_label = 1
            else:
                expected_label = 0
            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)

            if get_method(check_sum, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)

    def rule_simulation_R5(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        bins,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std, stength, is_vol_period

        def get_method(check_sum, expected_label, _method=None):
            if not enable_check_sum:
                check_sum = expected_label
            if _method == "negative":
                return (check_sum == expected_label) and (expected_label == 0)
            elif _method == "positive":
                return (check_sum == expected_label) and (expected_label == 1)
            else:  # bots
                return check_sum == expected_label

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0:
            t_r_c_d(cnt, False, 2)
        else:
            # a = data_y[peak_indexes]
            # b = data_y[valley_indexes]
            d = np.diff(np.log(raw_vals[cnt - 6 : cnt + 1]))
            # cut_off =  a.shape[0] if a.shape[0] <= b.shape[0] else b.shape[0]
            # if cut_off > 10:
            #     cut_off = 10
            # c = a[-cut_off:] - b[-cut_off:]
            # std = c.std(ddof=1)
            # std = a.std(ddof=1)
            time_diff = 3
            std = d.std(ddof=1)
            raw_return_current = ((raw_vals[cnt-6: cnt+1] - raw_vals[cnt-7: cnt]) / raw_vals[cnt-7: cnt]) * 100
            raw_return_prev = ((raw_vals[cnt-10: cnt-3] - raw_vals[cnt-11: cnt-4]) / raw_vals[cnt-11: cnt-4]) * 100
            raw_return_sum = np.sum(raw_return_current)
            std = raw_return_current.std(ddof=1) - raw_return_prev.std(ddof=1)
            std = np.abs(raw_return_current.std(ddof=1)) - np.abs(raw_return_prev.std(ddof=1))
            

            support_1 = np.diff(data_y[peak_indexes[-3:]])
            support_2 = np.diff(data_y[valley_indexes[-3:]])
            if ((support_1[0] * support_1[1]) < 0) or ((support_2[0] * support_2[1]) < 0):
                save_section = True
            if std > 1:
                is_vol_period = True
                # bins = 1
            # else:
                # if np.any(data_y[peak_indexes[-1]] == np.max(data_y[peak_indexes])):
                #     enable_check_sum = False
                #     save_section = True
                #     peak_path = np.append(data_y[peak_indexes[-1]], raw_vals[cnt])
                #     if np.sum(np.diff(peak_path)) > 0:
                #         expected_label = 1
                #     else:
                #         expected_label = 0
                # elif np.any(data_y[valley_indexes[-1]] == np.min(data_y[valley_indexes])):
                #     enable_check_sum = False
                #     save_section = True
                #     valley_path = np.append(data_y[valley_indexes[-1]], raw_vals[cnt])
                #     if np.sum(np.diff(valley_path)) > 0:
                #         expected_label = 1
                #     else:
                #         expected_label = 0
                # else:
                #     peak_path = np.append(data_y[peak_indexes[-bins:]], raw_vals[cnt])
                #     valley_path = np.append(data_y[valley_indexes[-bins:]], raw_vals[cnt])
                #     if np.sum([np.sum(np.diff(peak_path)), np.sum(np.diff(valley_path))]) > 0:
                #         expected_label = 1
                #     else:
                #         expected_label = 0

            peak_data = data_y[peak_indexes[-bins:]]
            valley_data = data_y[valley_indexes[-bins:]]
            if not (np.any(np.isin(peak_data, raw_vals[cnt - 1]))) or (np.any(np.isin(valley_data, raw_vals[cnt - 1]))):
                if raw_vals[cnt - 1] < raw_vals[cnt]:
                    add_valley = raw_vals[cnt - 1]
                    peak_path = np.append(peak_data, raw_vals[cnt])
                    valley_path = np.append(np.append(valley_data, add_valley), raw_vals[cnt])
                else:
                    add_peak = raw_vals[cnt - 1]
                    peak_path = np.append(np.append(peak_data, add_peak), raw_vals[cnt])
                    valley_path = np.append(valley_data, raw_vals[cnt])
            else:
                peak_path = np.append(peak_data, raw_vals[cnt])
                valley_path = np.append(valley_data, raw_vals[cnt])
            
            stength = np.sum([np.sum(np.diff(np.log(peak_path))), np.sum(np.diff(np.log(valley_path)))])
            if stength > 0:
                expected_label = 1
            else:
                expected_label = 0
            
                

            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)
            # check_sum = np.where(np.sum(raw_vals[cnt-3:cnt+1]) < 0, 0, 1)

            if get_method(check_sum, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)
    
    def rule_simulation_R5_KOSPI_BEST(
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        bins,
        plt,
        wrong_sample=False,
        enable_check_sum=False,
        method=None
    ):
        global save_section, std

        def get_method(check_sum, expected_label, _method=None):
            if not enable_check_sum:
                check_sum = expected_label
            if _method == "negative":
                return (check_sum == expected_label) and (expected_label == 0)
            elif _method == "positive":
                return (check_sum == expected_label) and (expected_label == 1)
            else:  # bots
                return check_sum == expected_label

        # Define Conditions
        if len(peak_indexes) == 0 or len(valley_indexes) == 0:
            t_r_c_d(cnt, False, 2)
        else:
            std = data_y[peak_indexes].std(ddof=1)
            if (data_y[peak_indexes[-1]] == np.max(data_y[peak_indexes])) or (data_y[valley_indexes[-1]] == np.min(data_y[valley_indexes])):
                save_section = True
                bins = 1
            peak_path = np.append(data_y[peak_indexes[-bins:]], raw_vals[cnt])
            valley_path = np.append(data_y[valley_indexes[-bins:]], raw_vals[cnt])
            if np.sum([np.sum(np.diff(peak_path)), np.sum(np.diff(valley_path))]) > 0:
                expected_label = 1
            else:
                expected_label = 0
            check_sum = np.where(raw_vals[cnt] - raw_vals[cnt - 1] < 0, 0, 1)

            if get_method(check_sum, expected_label, _method=method):
                performance_samples.append(expected_label == label)
                cul_performance.append(
                    round(np.sum(performance_samples) / len(performance_samples), 3)
                )
                plot(plt, expected_label, label, returns, wrong_sample)
                t_r_c_d(cnt, True, expected_label)
            else:
                t_r_c_d(cnt, False, 2)

    locals()["rule_simulation_R{}".format(str(t_e))](
        peak_indexes,
        valley_indexes,
        raw_vals,
        cnt,
        forward,
        bins,
        plt,
        wrong_sample=wrong_sample,
        enable_check_sum=enable_check_sum,
        method=method
    )


class variable_wrapper:
    def __init__(self, m_dict):
        self.__exp = m_dict

    def __getattr__(self, name):
        try:
            setattr(self, name, self.__exp[name])
            return self.__exp[name]
        except KeyError:
            setattr(self, name, None)
            return None

    def print(self):
        local_vars = [
            attr
            for attr in self.__dict__
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        ]
        for k in local_vars:
            print("{}: {}".format(k, self.__dict__[k]))


def std_return_plot(store_dir):
    data = pd.read_csv("{}/expected_returns_std.csv".format(store_dir))

    std = data["expected_return_std"]

    m_min = np.min(std)
    m_max = np.max(std)
    bins = np.arange(m_min, m_max, 1)

    y_data = list()
    z_data = list()
    x_data = list()
    for idx in range(len(bins)):
        if idx == 0:
            pass
        else:
            where_rows = (bins[idx - 1] < data["expected_return_std"]) & (data["expected_return_std"] < bins[idx])
            y_data.append(np.sum(data[where_rows]['expected_return']))
            z_data.append(np.sum(data[where_rows]['return']))
            x_data.append(bins[idx])

    # Plot main graph.
    (fig, ax) = plt.subplots()
    ax.plot(x_data, y_data, color="red", label="expected")
    ax.plot(x_data, z_data, color="black", label="real")
    plt.savefig("{}/std_return.png".format(store_dir))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--t_e", type=int, default=4)
    parser.add_argument("--mv", type=int, default=1)
    parser.add_argument("--bins", type=int, default=7)
    # Possible parameters: 'INX, KS200, XAU, US10YT, FTSE, GDAXI, SSEC, BVSP, N225, GB10YT, DE10YT, KR10YT, CN10YT, JP10YT, BR10YT'
    parser.add_argument("--target", type=str, default='INX')
    args = parser.parse_args()

    """
    Set Local variables
    """
    exp = {
        "t_e": 3,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 20,
        "wrong_sample": False,
        "enable_check_sum": False,
        "score": -57,
    }
    exp = {
        "t_e": 1,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 1,
        "wrong_sample": False,
        "enable_check_sum": True,
        "score": 49.678,
    }

    exp = {
        "t_e": 4,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 1,
        "bins": 3,
        "wrong_sample": False,
        "enable_check_sum": True,
        "save_section_eable": False,
        "score": 102,
    }
    exp = {
        "t_e": 4,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 1,
        "bins": 3,
        "wrong_sample": False,
        "enable_check_sum": False,
        "save_section_eable": False,
        "score": 191,
    }
    exp = {
        "t_e": 5,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 1,
        "bins": 3,
        "wrong_sample": False,
        "enable_check_sum": False,
        "save_section_eable": False,
        "score": 131,
    }

    # Seleted Paramaters [t_e: 5]
    exp = {
        "t_e": 5,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 1,
        "bins": 7,
        "wrong_sample": False,
        "enable_check_sum": True,
        "save_section_eable": False,
        "score": 257,
    }
    exp = {
        "t_e": 5,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 3,
        "bins": 3,
        "wrong_sample": False,
        "enable_check_sum": True,
        "save_section_eable": False,
        "score": 252,
    }
    exp = {
        "t_e": 5,
        "forward": 1,
        "method": "both",
        "canverse": 60,
        "mv": 10,
        "bins": 1,
        "wrong_sample": False,
        "enable_check_sum": True,
        "save_section_eable": False,
        "score": 212,
    }

    if args.t_e is not None:
        exp = {
            "t_e": int(args.t_e),
            "forward": 1,
            "method": "both",
            "canverse": 60,  # KS: 60, 
            "mv": int(args.mv),
            "bins": int(args.bins),
            "wrong_sample": False,
            "enable_check_sum": False,
            "save_section_eable": False,
            "score": 0,
        }
    exp = variable_wrapper(exp)

    global performance_samples, realized_returns, expected_returns, cul_performance, \
        data_x_s, data_y_s, peak_indexes, valley_indexes, mark_x_s, mark_y_s, mark_x_s_prediction, mark_y_s_prediction, \
            wrong_sample_cnt, save_section_eable, save_section, start_idx, end_idx, TradeDate, \
                target_name, std, expected_returns_std, canverse, stength, is_vol_period

    target_name = args.target
    save_dir = "{}_{}_{}_{}_{}".format(target_name, exp.t_e, exp.forward, exp.mv, exp.bins)
    file_name = "./datasets/rawdata/index_data/gold_index.csv"
    store_dir = "./temp/{}".format(save_dir)
    data = pd.read_csv(file_name)
    TradeDate = data["TradeDate"].values
    raw_vals = data[target_name].values
    is_vol_period = False
    
    # TradeDate = TradeDate[:150]
    # raw_vals = raw_vals[:150]

    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)

    t_e, forward, method, canverse, mv = (
        exp.t_e,
        exp.forward,
        exp.method,
        exp.canverse,
        exp.mv,
    )
    cnt = exp.canverse + 5
    bins = exp.bins
    wrong_sample = exp.wrong_sample
    enable_check_sum = exp.enable_check_sum
    exp.start_idx = exp.canverse - 1
    exp.end_idx = len(TradeDate) - (exp.canverse - 1)
    start_idx = exp.start_idx
    end_idx = exp.end_idx
    start_label = TradeDate[start_idx]
    end_label = TradeDate[end_idx]
    exp.target = target_name
    exp.print()

    """
    Logic section
    """
    if mv > 1:
        vals = rolling_apply(fun_mean, raw_vals, mv)
    else:
        vals = raw_vals

    wrong_sample_cnt = 0
    save_section_eable = exp.save_section_eable

    (
        performance_samples,
        realized_returns,
        expected_returns,
        cul_performance,
        total_return_chart_data,
        historical_return,
        expected_returns_std,
    ) = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )

    cnt = -1
    for i in range(len(data["TradeDate"].values)):
        try:
            save_section = False
            cnt = cnt + 1
            
            data_x = TradeDate[cnt - canverse : cnt]
            data_y = vals[cnt - canverse : cnt]

            data_x_s = TradeDate[cnt - canverse : cnt + canverse]
            data_y_s = raw_vals[cnt - canverse : cnt + canverse]
            mark_x_s = TradeDate[cnt]
            mark_y_s = raw_vals[cnt]
            mark_x_s_prediction = TradeDate[cnt + forward]
            mark_y_s_prediction = raw_vals[cnt + forward]

            # Find peaks(max).
            peak_indexes = signal.argrelextrema(data_y, np.greater)
            peak_indexes = peak_indexes[0]

            # Find valleys(min).
            valley_indexes = signal.argrelextrema(data_y, np.less)
            valley_indexes = valley_indexes[0]

            # make decision
            rule_simulation(
                peak_indexes,
                valley_indexes,
                raw_vals,
                cnt,
                forward,
                bins,
                plt,
                wrong_sample=wrong_sample,
                enable_check_sum=enable_check_sum,
                t_e=t_e,
                method=method,
            )

        except IndexError:
            plt.close()
            pass
    if wrong_sample:
        print("Count wrong samples: {}".format(wrong_sample_cnt))

    total_return_chart_plot(plt)
    pd.DataFrame(
        data=historical_return,
        columns=["real_return", "expected_return", "expected_label", "cnt"],
    ).to_csv("./{}/historical_return.csv".format(store_dir), index=None)

    pd.DataFrame(
        data=expected_returns_std,
        columns=["expected_return", "return", "expected_return_std"],
    ).to_csv("./{}/expected_returns_std.csv".format(store_dir), index=None)

    std_return_plot(store_dir)

    animate_gif("./{}".format(store_dir), duration=600)
    exit(1)
