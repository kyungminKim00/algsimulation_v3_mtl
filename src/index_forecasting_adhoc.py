from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""
import cython
import header.index_forecasting.RUNHEADER as RUNHEADER
import util
import datetime
from scipy.stats import entropy
import numpy as np
import pandas as pd
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

from sklearn.metrics import f1_score


@cython.ccall
def load(filepath, method):
    with open(filepath, "rb") as fs:
        if method == "pickle":
            data = pickle.load(fs)
    fs.close()
    return data


@cython.ccall
def save(filepath, method, obj):
    with open(filepath, "wb") as fs:
        if method == "pickle":
            pickle.dump(obj, fs)
    fs.close()


@cython.cfunc
@cython.locals(file_name=cython.p_char, process_id=cython.int, mode=cython.char)
@cython.returns(cython.p_char)
def recent_procedure(file_name, process_id, mode):
    json_file_location = ""
    with open("{}{}.txt".format(file_name, str(process_id)), mode) as _f_out:
        if mode == "w":
            print(RUNHEADER.m_name, file=_f_out)
        elif mode == "r":
            json_file_location = _f_out.readline()
        else:
            assert False, "<recent_procedure> : mode error"
        _f_out.close()
    return json_file_location.replace("\n", "")


def convert_pickable(RUNHEADER):
    keys = [
        key
        for key in RUNHEADER.__dict__.keys()
        if type(RUNHEADER.__dict__[key]) in [int, str, float, bool]
    ]
    _dict = [[element, RUNHEADER.__dict__[element]] for element in keys]
    return dict(_dict)


@cython.ccall
@cython.returns(cython.int)
@cython.locals(target_name=cython.int, forward_ndx=cython.int)
def test(a, b):
    return a + b


@cython.ccall
@cython.returns(cython.void)
@cython.locals(target_name=cython.p_char, forward_ndx=cython.p_char)
def print_confidence_performance(target_name, forward_ndx):
    stacks = None
    idxs = None
    base_dir = "./save/result/selected"
    loc = [
        it
        for it in os.listdir(base_dir)
        if target_name in it and "T" + forward_ndx in it
    ]
    for f_dir in loc:
        try:
            dirs = "{}/{}/final".format(base_dir, f_dir)
            for f_name in os.listdir(dirs):
                if "AC_Adhoc" in f_name:
                    if stacks is None:
                        data = pd.read_csv("{}/{}".format(dirs, f_name), index_col=0)
                        stacks = data.values
                        keys = data.keys()
                        idxs = dict(zip(keys, range(len(keys))))
                    else:
                        data = pd.read_csv("{}/{}".format(dirs, f_name), index_col=0)
                        stacks = np.concatenate([stacks, data.values], axis=0)
                    print(
                        "stack {} to calculate overall confidence performance".format(
                            dirs
                        )
                    )
        except FileNotFoundError:
            pass

    # f_name = loc
    # if stacks is None:
    #     data = pd.read_csv('{}'.format(f_name), index_col=0)
    #     stacks = data.values
    #     keys = data.keys()
    #     idxs = dict(zip(keys, range(len(keys))))
    # else:
    #     data = pd.read_csv('{}/{}'.format(dirs, f_name), index_col=0)
    #     stacks = np.concatenate([stacks, data.values], axis=0)

    summary = list()

    # bins = [[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6],
    #         [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    # for it in bins:
    #     tmp = stacks[:, idxs['P_Confidence']]
    #     search_idx = np.where(np.logical_and(tmp >= it[0], tmp <= it[1]))[0].tolist()
    #
    #     extracted_data_in_condidion = stacks[search_idx]
    #     result = f1_score(extracted_data_in_condidion[:, idxs['20days']],
    #                       extracted_data_in_condidion[:, idxs['P_20days']], average='weighted')
    #     summary.append([it[0], it[1], result])

    bins = np.arange(0, 1, 0.1)
    for it in bins:
        tmp = stacks[:, idxs["P_Confidence"]]
        search_idx = np.argwhere(tmp >= it).squeeze().tolist()

        extracted_data_in_condidion = stacks[search_idx]
        result = f1_score(
            extracted_data_in_condidion[:, idxs["20days"]].tolist(),
            extracted_data_in_condidion[:, idxs["P_20days"]].tolist(),
            average="weighted",
        )
        summary.append([it, result])

    plt.plot(np.array(summary)[:, 0], np.array(summary)[:, -1])
    plt.xticks(np.arange(0, 1, 0.1))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy(F1 score)")
    plt.title("Accuracy calibration")
    plt.grid(True)
    plt.savefig(
        "./save/result/selected/{}_T{}_confidence_calibration.jpeg".format(
            target_name, forward_ndx
        ),
        format="jpeg",
        dpi=600,
    )
    plt.close()


class Script:
    def __init__(
        self,
        result=None,
        model_location=None,
        f_base_model=None,
        f_model=None,
        adhoc_file=None,
        infer_mode=False,
        info=None,
        b_naive=True,
        performed_date=None,
    ):
        self.f_base_model = f_base_model
        self.f_model = f_model
        self.result = result
        self.model_location = model_location
        self.adhoc_file = adhoc_file
        self.pl = 2.5
        self.vl = 2
        self.ev = 0.8
        self.infer_mode = infer_mode
        self.info = info
        self.b_naive=b_naive
        self.performed_date = performed_date

    def align_consistency(self, data):
        tmp_dict = {0: "P_return", 1: "P_return2"}
        for idx in range(len(data)):
            min_idx = np.argmin([data["P_return"][idx], data["P_return2"][idx]])
            max_idx = np.argmax([data["P_return"][idx], data["P_return2"][idx]])

            if data["P_20days"][idx] == 1:
                if data[tmp_dict[max_idx]][idx] >= 0:
                    pass
                else:
                    data[tmp_dict[max_idx]][idx] = np.abs(data[tmp_dict[max_idx]][idx])
            else:
                if data[tmp_dict[min_idx]][idx] < 0:
                    pass
                else:
                    data[tmp_dict[min_idx]][idx] = -1 * data[tmp_dict[min_idx]][idx]
        return data

    def get_dates(self, dates, forward):
        base_dates = list()
        for i in range(len(dates)):
            cnt = 0
            datetime_obj = datetime.datetime.strptime(dates[i], "%Y-%m-%d")
            while True:
                datetime_obj += datetime.timedelta(days=-1)
                if datetime_obj.weekday() < 5:  # keep working days
                    cnt = cnt + 1
                    if forward == cnt:
                        datetime_obj = datetime_obj.strftime("%Y-%m-%d")
                        base_dates.append(datetime_obj)
                        break
        assert len(base_dates) == len(dates)

        return base_dates

    def reporting(self, data, file_name, info):
        g_date = self.performed_date
        if g_date is None:
            g_date = "{}-{}-{}".format(
            info.split("_")[2][:4], info.split("_")[2][4:6], info.split("_")[2][6:8]
        )

        col_names = [
            "seq_num",
            "mrkt_cd",
            "forward",
            "performed_date",
            "base_date",
            "base_index",
            "prediction_date",
            "prediction_index_min",
            "prediction_index_max",
            "prediction_return_min",
            "prediction_return_max",
            "prediction_up_down",
            "confidence",
        ]
        cnt_cols = len(col_names)
        cnt_rows = data.shape[0]
        seq_num = np.arange(cnt_rows)
        mrkt_cd = np.array([info.split("_")[0] for i in range(cnt_rows)])
        forward = np.array([info.split("_")[1][1:] for i in range(cnt_rows)])
        performed_date = np.array([g_date for i in range(cnt_rows)])
        base_date = np.array(
            self.get_dates(data["P_Date"], int(info.split("_")[1][1:]))
        )
        prediction_index_min = np.min([data["P_index"], data["P_index2"]], axis=0)
        prediction_index_max = np.max([data["P_index"], data["P_index2"]], axis=0)
        prediction_return_min = np.min([data["P_return"], data["P_return2"]], axis=0)
        prediction_return_max = np.max([data["P_return"], data["P_return2"]], axis=0)

        report = (
            np.concatenate(
                [
                    seq_num,
                    mrkt_cd,
                    forward,
                    performed_date,
                    base_date,
                    data["today_index"],
                    data["P_Date"],
                    prediction_index_min,
                    prediction_index_max,
                    prediction_return_min,
                    prediction_return_max,
                    data["P_20days"],
                    data["P_Confidence"],
                ]
            )
            .reshape(cnt_cols, cnt_rows)
            .T
        )
        pd.DataFrame(data=report, columns=col_names).to_csv(file_name, index=None)

        model_info = self.model_location + ".txt"
        fp = open(model_info, "w")
        fp.write(self.f_model)
        fp.close()

    def run_adhoc(self):
        # get model list for evaluate performance
        models = os.listdir(self.model_location)

        filenames = list()
        [filenames.append(_model) for _model in models if ".csv" in _model]

        # calculate confidence and probability
        if self.b_naive:
            candidate_models = self.naive_filter(filenames)
        else:
            candidate_models = filenames
            
        if len(candidate_models) == 0:
            print("Skip ad-hoc process - there is no proper candidate models")
            sys.exit()

        selected_model = self.final_model(filenames)
        date, example, confidence, probability = self.calculate(
            candidate_models, selected_model
        )

        # get final result
        s_model_result = "{}/{}".format(self.model_location, self.f_model)

        # apply adhoc process
        data = pd.read_csv(s_model_result, index_col=[0])
        data["P_Confidence"] = confidence
        data["P_Probability"] = probability
        data = self.align_consistency(data)
        pd.DataFrame(data=data).to_csv("{}/{}".format(self.result, self.adhoc_file))

        if self.infer_mode:
            file_name = (
                "/".join(self.result.split("/")[:-2])
                + "/prediction_with_confidence_and_adhoc_process.csv"
            )
            self.reporting(data, file_name, self.info)

    def _calculate(self, examples):
        up_p = np.sum(examples, axis=0) / examples.shape[0]
        down_p = 1 - up_p
        shanon_entropy = entropy([down_p, up_p], base=2)
        confidence = 1 - shanon_entropy

        assert (
            len(np.argwhere(shanon_entropy > 1)) + len(np.argwhere(shanon_entropy < 0))
            == 0
        ), "Error check your logic"
        assert (
            len(np.argwhere(confidence > 1)) + len(np.argwhere(confidence < 0)) == 0
        ), "Error check your logic"

        u_confidence = np.empty([examples.shape[1]])
        d_confidence = np.empty([examples.shape[1]])
        for idx in range(examples.shape[1]):
            if down_p[idx] > 0.5:
                d_confidence[idx] = confidence[idx]
                u_confidence[idx] = 1 - confidence[idx]
            else:
                d_confidence[idx] = 1 - confidence[idx]
                u_confidence[idx] = confidence[idx]
        return d_confidence, u_confidence, down_p, up_p

    def calculate(self, candidate_models, selected_model):
        # selected_model_result, candidate_model_result = \
        #     self.get_model_results_selected(selected_model), self.get_model_results_candidate(candidate_models)
        selected_model_result, candidate_model_result = [
            selected_model
        ], candidate_models

        # calculate confidence for up and down
        s_date, s_examples = self.collect_examples(selected_model_result)
        c_date, c_examples = self.collect_examples(candidate_model_result)

        d_confidence, u_confidence, down_p, up_p = self._calculate(c_examples)

        assert s_examples.shape[0] == 1, "selected model would be 1 for now"
        s_examples = s_examples.squeeze()
        confidence = np.zeros(len(s_date))
        probability = np.zeros(len(s_date))
        d_idx, u_idx = [], []
        for cond in [0, 1]:
            array_idx = np.argwhere(s_examples == cond)
            if len(array_idx) > 0:
                idx = array_idx.squeeze() if len(array_idx) > 1 else array_idx[0]
                if cond == 0:
                    d_idx = idx
                    c = d_confidence
                    p = down_p
                else:
                    u_idx = idx
                    c = u_confidence
                    p = up_p
                confidence[idx] = c[idx]
                probability[idx] = p[idx]

        assert len(d_idx) + len(u_idx) == len(s_examples), "check up & down index"
        
        return (
            s_date.tolist(),
            s_examples.tolist(),
            confidence.tolist(),
            probability.tolist(),
        )

    def collect_examples(self, result_file):
        b_shape, date, examples = None, None, None
        for filename in result_file:
            data = pd.read_csv("{}/{}".format(self.model_location, filename))
            key_dict = data.keys()

            # date and target result (up & down for target date)
            target_date, target_result = key_dict[1], key_dict[4]
            value = np.expand_dims(data[target_result].values, axis=0)
            # value = data[target_result].values
            value = np.where(value >= 0, 1, 0)
            if b_shape is None:
                date = data[target_date].values
                examples = value
                b_shape = True
            else:
                examples = np.append(examples, value, axis=0)
        return date, examples

    def naive_filter(self, model_list):
        filtered_model = list()
        for model_name in model_list:
            token = model_name.split("_")
            try:
                if (
                    float(token[5][2:]) < self.pl
                    and float(token[6][2:]) < self.vl
                    and float(token[7][2:]) > self.ev
                ):
                    filtered_model.append(model_name)
            except IndexError:
                filtered_model.append(model_name)
        assert len(filtered_model), "All model are denied from the naive filter"

        return sorted(list(set(filtered_model)), key=len)

    def final_model(self, model_lists):
        for item in model_lists:
            if self.f_model in item:
                return item
        return None

    def get_model_results_candidate(self, models):
        model_result, models_results = list(), list()
        models_result_loc = os.listdir(self.model_location)
        [
            models_results.append(_model)
            for _model in models_result_loc
            if ".csv" in _model
        ]

        for _model in models:
            for _model_result in models_results:
                if (
                    int(_model.split("_")[4]) == int(_model_result.split("_")[4])
                    and float(_model.split("_")[6][2:])
                    == float(_model_result.split("_")[5][2:])
                    and float(_model.split("_")[7][2:])
                    == float(_model_result.split("_")[6][2:])
                    and float(_model.split("_")[8][2:-4])
                    == float(_model_result.split("_")[7][2:])
                ):
                    model_result.append(_model_result)
        return model_result

    def get_model_results_selected(self, models):
        model_result, models_results = list(), list()
        models_result_loc = os.listdir(self.model_location)
        [
            models_results.append(_model)
            for _model in models_result_loc
            if ".csv" in _model
        ]

        for _model in models:
            for _model_result in models_results:
                if (
                    int(_model.split("_")[4]) == int(_model_result.split("_")[4])
                    and float(_model.split("_")[6][2:])
                    == float(_model_result.split("_")[5][2:])
                    and float(_model.split("_")[7][2:])
                    == float(_model_result.split("_")[6][2:])
                    and float(_model.split("_")[8][2:-4])
                    == float(_model_result.split("_")[7][2:])
                ):
                    model_result.append(_model_result)
        return model_result


class Adhoc:
    def __init__(self, m_target_index=None, forward_ndx=None, dataset_version=None, performed_date=None):
        self.m_target_index = m_target_index
        self.forward_ndx = forward_ndx
        self.dataset_version = dataset_version
        self.target_name = RUNHEADER.target_id2name(self.m_target_index)
        self.result_dir = "./save/result/"
        # self.tDir_ = '{}_T{}_{}'.format(self.target_name, self.forward_ndx, self.dataset_version)
        self.tDir_ = "{}_T{}".format(self.target_name, self.forward_ndx)
        self.tDir = self.result_dir + "selected/" + self.tDir_ + "/final"
        self.adhoc_file = "AC_Adhoc.csv"
        self.performed_date = performed_date

    def run(self):
        """configuration"""
        f_base_model = ""
        f_model = ""
        for f_name in os.listdir(self.tDir):
            if "jpeg" not in f_name and "csv" not in f_name:
                f_base_model = f_name
            if "csv" in f_name:
                f_model = f_name[2:]

        if f_base_model != "" and f_model != "":
            _model_location = [
                fn for fn in os.listdir(self.result_dir) if f_base_model in fn
            ].pop()
            _model_location = "{}{}".format(self.result_dir, _model_location)

            # _model_location = './save/result/' + args.dataset_version + '/' + f_base_model
            _result = self.tDir

            """ run application
            """
            sc = Script(
                result=_result,
                model_location=_model_location,
                f_base_model=f_base_model,
                f_model=f_model,
                adhoc_file=self.adhoc_file,
                performed_date=self.performed_date,
            )
            pd.set_option("mode.chained_assignment", None)
            sc.run_adhoc()
            pd.set_option("mode.chained_assignment", "warn")

            # print test environments
            print("\nadhoc dataset: {}".format(self.tDir_))
            print("target loc: {}".format(self.tDir))
            return 1
        else:  # the prediction would be performed with a predefined base model
            print(
                "there is no final model, the prediction would be performed with a predefined base model"
            )
            print("Pass: {}".format(self.tDir_))
            return 0


def get_header_info(json_location):
    return util.json2dict("{}/agent_parameter.json".format(json_location))


def get_model_name(model_dir, model_name):
    search = "./save/model/rllearn/{}".format(model_dir)
    for it in os.listdir(search):
        if model_name in it:
            return it


def update_model_pool(m_target_index, forward_ndx, dataset_version, flag, init_repo_model=0):
    target_name = RUNHEADER.target_id2name(m_target_index)
    domain_detail = "{}_T{}_{}".format(target_name, forward_ndx, dataset_version)
    domain = "{}_T{}".format(target_name, forward_ndx)
    # src_dir = './save/result/selected/{}/final'.format(domain_detail)
    src_dir = "./save/result/selected/{}/final".format(domain)
    target_file = "./save/model_repo_meta/{}.pkl".format(domain)
    base_dir = None
    model_name = None
    time_now = (
        str(datetime.datetime.now())[:-10]
        .replace(":", "-")
        .replace("-", "")
        .replace(" ", "_")
    )

    if flag:  # exist a selected final model
        for it in os.listdir(src_dir):
            if "AC_Adhoc" not in it and "jpeg" not in it:
                if "csv" in it:
                    model_name = it
                else:
                    base_dir = it

        if base_dir is not None and model_name is not None:
            header = get_header_info("./save/model/rllearn/{}".format(base_dir))
            meta = {
                "domain_detail": domain_detail,
                "domain": domain,
                "src_dir": src_dir,
                "target_file": target_file,
                "base_dir": base_dir,
                "model_name": get_model_name(
                    base_dir,
                    "sub_epo_{}".format(
                        model_name.split("sub_epo_")[1:][0].split("_")[0]
                    ),
                ),
                "create_date": time_now,
                "target_name": header["target_name"],
                "m_name": header["m_name"],
                "dataset_version": header["dataset_version"],
                "m_offline_buffer_file": int(init_repo_model),
                "latest": True,
                "current_period": True,  # the best at the moment
            }
            if os.path.isfile(target_file):
                meta_list = load(target_file, "pickle")
                # mark latest model
                for idx in range(len(meta_list)):
                    meta_list[idx]["latest"] = False
                    meta_list[idx]["current_period"] = False
                meta_list.append(meta)
                save(target_file, "pickle", meta_list)
            else:
                meta_list = list()
                meta_list.append(meta)
                save(target_file, "pickle", meta_list)
    else:  # use base model for prediction
        if os.path.isfile(target_file):
            meta_list = load(target_file, "pickle")
            # get info from base models
            t_idx = None
            tmp_meta = None
            for idx in range(len(meta_list)):
                if meta_list[idx]["latest"]:
                    meta_list[idx]["latest"] = False
                    meta_list[idx]["current_period"] = False
                    tmp_meta = meta_list[idx]

            meta = {
                "domain_detail": domain_detail,
                "domain": domain,
                "src_dir": None,
                "target_file": target_file,
                "base_dir": tmp_meta["base_dir"],
                "model_name": tmp_meta["model_name"],
                "create_date": time_now,
                "target_name": target_name,
                "m_name": tmp_meta["m_name"],
                "dataset_version": dataset_version,
                "m_offline_buffer_file": tmp_meta["m_offline_buffer_file"],
                "latest": True,
                "current_period": False,  # historical best
            }
            meta_list.append(meta)
            save(target_file, "pickle", meta_list)
        else:
            assert False, "There is no predefined base model"
