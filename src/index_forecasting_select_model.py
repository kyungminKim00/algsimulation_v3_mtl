from __future__ import absolute_import, division, print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import argparse
import os
import pickle
import re
import shutil
import sys
from distutils.dir_util import copy_tree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

import header.index_forecasting.RUNHEADER as RUNHEADER
import util

matplotlib.use("agg")


class Data:
    def __init__(
        self,
        rDir=None,
        tDir=None,
        model_name=None,
        val_dir_return=None,
        val_dir_index=None,
        test_dir_return=None,
        test_dir_index=None,
        file_name=None,
        retry=None,
        max_cnt=None,
        th_m_score=None,
        th_dict=None,
        soft_cond_retry=None,
    ):
        self.rDir = rDir
        self.tDir = tDir
        self.max_cnt = max_cnt
        self.model_name = model_name
        self.id = self._get_id(file_name)
        self.file_name = file_name
        self.val_dir_return = val_dir_return
        self.val_dir_index = val_dir_index
        self.test_dir_return = test_dir_return
        self.test_dir_index = test_dir_index
        self.columns = self._get_columns()
        self.test_return_file = ""
        self.test_index_file = ""
        self.test_csv_file = self._get_test_csv()
        self._get_test_result(self.test_dir_return, self.id)
        self.retry = retry
        self.sub_m_score = None
        self.m_score = None
        self.m_avg_r_acc = None
        self.m_avg_train_c_acc = None
        # self.select_criteria = select_criteria
        self.soft_cond_retry = soft_cond_retry

        self.th_pl = th_dict["th_pl"]  # less control-parm
        self.th_vl = th_dict["th_vl"]  # less
        self.th_ev = th_dict["th_ev"]  # less
        self.th_v_c = th_dict["th_v_c"]  # more
        self.th_train_c_acc = th_dict["th_train_c_acc"]  # more  control-parm
        self.th_v_mae = th_dict["th_v_mae"]  # less
        self.th_v_r_acc = th_dict["th_v_r_acc"]  # more
        self.th_v_ev = th_dict["th_v_ev"]  # more control-parm
        self.th_sub_m_score = th_m_score[0]  # more
        self.th_m_score = th_m_score[1]  # more
        self.th_epoch = th_dict["th_epoch"]  # more

        self.selected = self.naive_filter()

    def _get_test_csv(self):
        loc = f"{self.rDir}/{self.model_name}"
        for f_name in os.listdir(loc):
            if ("csv" in f_name) and ("sub_epo_" + self.id in f_name):
                return f"{loc}/{f_name}"

    def _get_id(self, f_name):
        token = f_name.split("_")
        self.pl = token[5][2:]
        self.vl = token[6][2:]
        self.ev = token[7][2:]
        self.v_c = token[9]
        self.train_c_acc = token[12]
        self.v_mae = token[16]
        self.v_r_acc = token[17]
        self.v_ev = token[19][:-4]
        return token[4]

    def _get_test_result(self, test_dir, id):
        for f_name in os.listdir(test_dir):
            if "csv" in f_name:
                token = f_name.split("_")
                if id == token[4]:
                    self.t_c = token[9]
                    self.t_c_acc = token[12]
                    self.t_mae = token[16]
                    self.t_r_acc = token[17]
                    self.t_ev = token[19][:-4]
                    self.test_return_file = f"{self.test_dir_return}/marketname/{'_'.join(f_name.split('_')[:8])}"
                    self.test_index_file = None
                    break

    def tolist(self):
        return [
            self.model_name,
            self.id,
            self.t_c_acc,
            self.t_r_acc,
            self.t_c,
            self.pl,
            self.vl,
            self.ev,
            self.v_c,
            self.train_c_acc,
            self.v_mae,
            self.v_r_acc,
            self.v_ev,
            self.val_dir_return,
            self.val_dir_index,
            self.test_dir_return,
            self.test_dir_index,
            self.t_ev,
            self.test_return_file,
            self.test_index_file,
            self.tDir,
            self.test_csv_file,
            self.t_mae,
            self.sub_m_score,
            self.m_score,
            self.m_avg_r_acc,
            self.m_avg_train_c_acc,
        ]

    def _get_columns(self):
        return [
            "model_name",
            "id",
            "test_c_acc",
            "test_r_acc",
            "test_consistency",
            "train_pl",
            "train_vl",
            "train_ev",
            "validate_consistency",
            "train_c_acc",
            "validate_mae",
            "validate_r_acc",
            "validate_ev",
            "validate_dir_return",
            "validate_dir_index",
            "test_dir_return",
            "test_dir_index",
            "test_ev",
            "test_return_file",
            "test_index_file",
            "tDir",
            "test_csv_file",
            "test_mae",
            "sub_m_score",
            "m_score",
            "m_avg_r_acc",
            "m_avg_train_c_acc",
        ]

    def _status_print(self, model_name=None, print_str=None):
        b_print = True
        if b_print:  # global variables
            print(print_str.format(model_name))
            print(
                f"\tid:{self.id} pl:{self.pl}/{self.th_pl} vl:{self.vl}/{self.th_vl} ev:{self.ev}/{self.th_ev} v_c:{self.v_c}/{self.th_v_c} train_c_acc:{self.train_c_acc}/{self.th_train_c_acc} v_mae:{self.v_mae}/{self.th_v_mae} v_r_acc:{self.v_r_acc}/{self.th_v_r_acc} v_ev:{self.v_ev}/{self.th_v_ev}"
            )

    def naive_filter(self):
        if self.model_name.split("_")[5] == "v3":  # Disable v3 model
            return False
        if self.soft_cond_retry == 0 and self.retry == 0:
            if (
                float(self.vl) <= self.th_vl
                and float(self.pl) <= self.th_pl
                and float(self.ev) >= self.th_ev
                and float(self.v_c) >= self.th_v_c
                and float(self.train_c_acc) >= self.th_train_c_acc
                and float(self.v_mae) <= self.th_v_mae
                and float(self.v_r_acc) >= self.th_v_r_acc
                and float(self.v_ev) >= self.th_v_ev
            ):
                self._status_print(self.model_name, "\n [0 - {}] Init Rule Activated")
                return True

        elif self.soft_cond_retry == 1:
            if self.max_cnt >= self.retry:
                if (
                    float(self.vl) <= (self.th_vl + self.retry * 0.005)
                    and float(self.pl) <= (self.th_pl + self.retry * 0.01)
                    and float(self.ev) >= self.th_ev
                    and float(self.v_c) >= self.th_v_c
                    and float(self.train_c_acc)
                    >= (self.th_train_c_acc - self.retry * 0.005)
                    and float(self.v_mae) <= self.th_v_mae
                    and int(self.id) >= self.th_epoch
                    and float(self.v_r_acc) >= self.th_v_r_acc
                    and float(self.v_ev) >= (self.th_v_ev - self.retry * 0.03)
                ):
                    self._status_print(self.model_name, "\n [1 - {}] pick first best 1")
                    return True

        elif self.soft_cond_retry == 2:
            if self.max_cnt >= self.retry:
                if (
                    float(self.vl) <= (self.th_vl + self.retry * 0.005)
                    and float(self.pl) <= (self.th_pl + self.retry * 0.01)
                    and float(self.ev) >= self.th_ev
                    and float(self.v_c) >= self.th_v_c
                    and float(self.train_c_acc)
                    >= (self.th_train_c_acc - self.retry * 0.005)
                    and float(self.v_mae) <= self.th_v_mae
                    and int(self.id) >= self.th_epoch
                    and float(self.v_r_acc) >= (self.th_v_r_acc - self.retry * 0.005)
                    and float(self.v_ev) >= (self.th_v_ev - self.retry * 0.025)
                ):
                    self._status_print(self.model_name, "\n [2 - {}] pick first best 2")
                    return True

        elif self.soft_cond_retry == 3 and self.retry == 0:
            if (
                float(self.v_r_acc) >= 0.75
                and int(self.id) >= self.th_epoch
                and float(self.train_c_acc) >= 0.84
                and float(self.v_ev) > 0.1
            ):
                self._status_print(self.model_name, "\n [3 - {}] pick third best")
                return True

        elif self.soft_cond_retry == 4 and self.retry == 0:
            if (
                float(self.v_r_acc) >= 0.75
                and int(self.id) >= self.th_epoch
                and float(self.train_c_acc) >= 0.84
                and float(self.v_ev) > -0.1
            ):
                self._status_print(self.model_name, "\n [4 - {}] pick fourth best")
                return False  # Disable
                return True

        elif self.soft_cond_retry == 5 and self.retry == 0:
            self._status_print(
                self.model_name, "\n [5 - {}] pick fifth best (pool best)"
            )
            return False
        return False  # pick pool best model


class Script:
    def __init__(
        self,
        tDir=None,
        rDir=None,
        max_cnt=None,
        select_criteria=None,
        soft_cond=True,
        th_dict=None,
    ):
        self.tDir = tDir
        self.rDir = rDir
        self.base = "/".join(tDir.split("/")[:-1])

        self.gathered_results = None
        self.column_name = None
        self.max_cnt = max_cnt
        self.select_criteria = select_criteria
        self.soft_cond = soft_cond
        self.th_dict = th_dict
        self.th_epoch = th_dict["th_epoch"]
        self.th_sub_score = th_dict["th_sub_score"]
        self.pool_best = None

    def _sort(self, models):
        aa = list()
        bb = list()
        for it in models:
            if it.split("_")[5] == "v1":
                bb.append(it)
            else:
                aa.append(it)
        return aa + bb

    def fill_statistics_model(self, avg_criteria, dict_str):
        if dict_str == "validate_ev":
            for idx in range(self.gathered_results.shape[0]):
                for item in avg_criteria:
                    if (
                        self.gathered_results[idx, self.dict_col2idx["model_name"]]
                        == item[0]
                    ):  # model score by criteria
                        self.gathered_results[idx, self.dict_col2idx["m_score"]] = item[
                            2
                        ]

                        if (
                            4 <= idx <= (self.gathered_results.shape[0])
                        ):  # model score calculated by sub-score
                            self.gathered_results[
                                idx, self.dict_col2idx["sub_m_score"]
                            ] = np.mean(
                                np.array(
                                    self.gathered_results[
                                        idx - 4 : idx, self.dict_col2idx[dict_str]
                                    ],
                                    dtype=np.float,
                                )
                            )
                        else:
                            self.gathered_results[
                                idx, self.dict_col2idx["sub_m_score"]
                            ] = 0  # dummy data
        elif dict_str == "validate_r_acc":
            for idx in range(self.gathered_results.shape[0]):
                for item in avg_criteria:
                    if (
                        self.gathered_results[idx, self.dict_col2idx["model_name"]]
                        == item[0]
                    ):  # model score by criteria
                        self.gathered_results[
                            idx, self.dict_col2idx["m_avg_r_acc"]
                        ] = item[2]
        elif dict_str == "train_c_acc":
            for idx in range(self.gathered_results.shape[0]):
                for item in avg_criteria:
                    if (
                        self.gathered_results[idx, self.dict_col2idx["model_name"]]
                        == item[0]
                    ):  # model score by criteria
                        self.gathered_results[
                            idx, self.dict_col2idx["m_avg_train_c_acc"]
                        ] = item[2]

    def _gather_result_information(self, output=False, retry=0, soft_cond_retry=0):
        # get model list for evaluate performance
        self.item_container = []
        gathered_results = []
        column_name = None
        models = self.rDir
        models = self._sort(models)
        self.pool_best = None  # no proper model but it is a pool best model

        # for dir_name in models:
        for dir_name in models:
            # tmp_dir = "./save/result/" + dir_name
            # val_dir_return = tmp_dir + "/validation/return/"
            # val_dir_index = tmp_dir + "/validation/index/"
            # test_dir_return = tmp_dir + "/return/"
            # test_dir_index = tmp_dir + "/index/"

            tmp_dir = "./save/result/" + dir_name
            val_dir_return = tmp_dir + "/validation/"
            val_dir_index = None
            test_dir_return = tmp_dir
            test_dir_index = None

            # file validation
            r = re.compile(".*csv")
            r2 = re.compile(".*csv")

            try:
                if len(list(filter(r2.match, os.listdir(tmp_dir)))) > 0:
                    for file_name in os.listdir(val_dir_return):
                        if "csv" in file_name:
                            if int(file_name.split("_")[4]) >= self.th_dict["th_epoch"]:
                                item = Data(
                                    rDir="./save/result",
                                    tDir=self.tDir,
                                    model_name=dir_name,
                                    val_dir_return=val_dir_return,
                                    val_dir_index=val_dir_index,
                                    test_dir_return=test_dir_return,
                                    test_dir_index=test_dir_index,
                                    file_name=file_name,
                                    retry=retry,
                                    max_cnt=self.max_cnt,
                                    th_m_score=[
                                        self.th_sub_score,
                                        self.select_criteria,
                                    ],
                                    th_dict=self.th_dict,
                                    soft_cond_retry=soft_cond_retry,
                                )
                                self.item_container.append(item)
                                gathered_results.append(item.tolist())
                                column_name = item.columns
            except Exception as e:
                pass
        self.column_name = column_name
        self.dict_col2idx = dict(
            list(zip(self.column_name, range(len(self.column_name))))
        )
        self.gathered_results = np.array(gathered_results)
        assert len(self.column_name) == len(gathered_results[0]), "lens are different"

        #  data fill and restore pool best models
        avg_criteria = self.avg_criteria(gathered_results, "validate_ev")
        pool_pick_opt0 = avg_criteria[np.argwhere(avg_criteria[:, -1] > 0)][
            :, 0, 0
        ].tolist()
        self.fill_statistics_model(avg_criteria, "validate_ev")

        avg_criteria = self.avg_criteria(gathered_results, "validate_r_acc")
        pool_pick_opt1 = avg_criteria[np.argwhere(avg_criteria[:, -1] >= 0.60)][
            :, 0, 0
        ].tolist()
        self.fill_statistics_model(avg_criteria, "validate_r_acc")

        avg_criteria = self.avg_criteria(gathered_results, "train_c_acc")
        pool_pick_opt2 = avg_criteria[np.argwhere(avg_criteria[:, -1] >= 0.82)][
            :, 0, 0
        ].tolist()
        self.fill_statistics_model(avg_criteria, "train_c_acc")

        pool_pick_opt = pool_pick_opt0 + pool_pick_opt1 + pool_pick_opt2
        cnt = 0
        for it in pool_pick_opt:
            if cnt < pool_pick_opt.count(it) and pool_pick_opt.count(it) > 1:
                cnt = pool_pick_opt.count(it)
                self.pool_best = it

        # save output
        if output:
            pd.DataFrame(data=self.gathered_results, columns=self.column_name).to_csv(
                self.tDir + "/gathered_results.csv"
            )

        # fill self.item_container
        assert (
            len(self.item_container) == self.gathered_results.shape[0]
        ), "should be the same"

        for i, _ in enumerate(self.item_container):
            self.item_container[i].sub_m_score = self.gathered_results[i][
                self.dict_col2idx["sub_m_score"]
            ]
            self.item_container[i].m_score = self.gathered_results[i][
                self.dict_col2idx["m_score"]
            ]
            self.item_container[i].m_avg_r_acc = self.gathered_results[i][
                self.dict_col2idx["m_avg_r_acc"]
            ]
            self.item_container[i].m_avg_train_c_acc = self.gathered_results[i][
                self.dict_col2idx["m_avg_train_c_acc"]
            ]

    def avg_criteria(self, data, criteria):  # calculate model score
        criteria = self.dict_col2idx[criteria]
        model_name = list(set(np.array(data)[:, self.dict_col2idx["model_name"]]))
        avg_matrix = np.zeros([len(model_name), 3], dtype=np.object)
        avg_matrix[:, self.dict_col2idx["model_name"]] = model_name

        for idx in range(avg_matrix.shape[0]):
            for item in data:
                if (
                    avg_matrix[idx][0] == item[self.dict_col2idx["model_name"]]
                ) and int(item[self.dict_col2idx["id"]]) >= self.th_epoch:
                    avg_matrix[idx, 1] = avg_matrix[idx, 1] + float(item[criteria])
                    avg_matrix[idx, 2] = avg_matrix[idx, 2] + 1
        avg_matrix[:, 2] = np.divide(avg_matrix[:, 1], avg_matrix[:, 2])
        return avg_matrix

    def printout_model_info(self, m_info):
        colname = [
            "model_name",
            "id",
            "test_c_acc",
            "test_r_acc",
            "test_consistency",
            "train_pl",
            "train_vl",
            "train_ev",
            "validate_consistency",
            "train_c_acc",
            "validate_mae",
            "validate_r_acc",
            "validate_ev",
            "validate_dir_return",
            "validate_dir_index",
            "test_dir_return",
            "test_dir_index",
            "test_ev",
            "test_return_file",
            "test_index_file",
            "tDir",
            "test_csv_file",
            "test_mae",
            "sub_m_score",
            "m_score",
            "m_avg_r_acc",
            "m_avg_train_c_acc",
        ]

        f_name = f"{m_info[self.dict_col2idx['tDir']]}/final/{m_info[self.dict_col2idx['model_name']]}"
        fp = open(f_name, "w")
        for col_name in range(len(colname)):
            fp.write(f"{col_name} : {m_info[col_name]}")
        fp.close()

        # 컨피던스 Score 사용시는 아래코드 삭제 하여야 함. 컨피던스 스코어가 유의미하지 않고 속도 및 저장공간 이슈로 최종 모형외 삭제 함
        if RUNHEADER._debug_on:
            pass
        else:
            for rm_file in os.listdir(
                f"./save/model/rllearn/{m_info[self.dict_col2idx['model_name']]}"
            ):
                if ".pkl" in rm_file:
                    if f"sub_epo_{str(m_info[self.dict_col2idx['id']])}" in rm_file:
                        pass
                    else:
                        os.remove(
                            f"./save/model/rllearn/{m_info[self.dict_col2idx['model_name']]}/{rm_file}"
                        )

    def post_decision(self, selected_model):
        criteria_list = []
        for model in selected_model:
            base_dir = "/".join(
                model[self.dict_col2idx["validate_dir_return"]].split("/")[:-2]
            )
            for fn in os.listdir(base_dir):
                if "sub_epo_" + model[self.dict_col2idx["id"]] in fn and ".csv" in fn:
                    data = pd.read_csv(base_dir + "/" + fn)
                    criteria_list.append(
                        (
                            np.corrcoef(data["Return"][-10:], data["P_return"][-10:])[
                                0, 1
                            ]
                        )
                    )
        if len(criteria_list) > 0:
            # return np.argmin(criteria_list)
            return np.argmax(criteria_list)
        else:
            return None

    def run_s_model(self, dset_v=None, b_batch_test=False):
        b_exit = False
        retry = 0
        soft_cond_retry = 0
        while not b_exit:
            sys.stdout.write(f"\r>> Adjusting parameters: {retry}")
            sys.stdout.flush()

            # copy candidate model to
            selected_model = []
            m_pass = False
            self._gather_result_information(True, retry, soft_cond_retry)
            for item in self.item_container:
                if soft_cond_retry == 5:  # pick pool best .. get reasonable models
                    if (
                        item.model_name == self.pool_best
                        and float(item.v_r_acc) >= 0.65
                        and float(item.train_c_acc) >= 0.82
                        and int(item.id) >= item.th_epoch
                        and float(item.pl) <= 1.88
                        and float(item.vl) <= 1.8
                        and float(item.v_ev) > 0.1
                    ):
                        item.selected = True
                        m_pass = True
                else:
                    # model statistics test
                    if (
                        (item.selected is True)
                        and (float(item.sub_m_score) >= item.th_sub_m_score)
                        and (float(item.m_score) >= item.th_m_score)
                        and (int(item.id) >= item.th_epoch)
                    ):
                        m_pass = True
                    if item.selected == 99:  # alternative rule - Disabled
                        m_pass = True

                # m_pass = True
                if m_pass:
                    selected_model.append(item.tolist())
                    for market_name in list(RUNHEADER.mkidx_mkname.values()):
                        if market_name != "TOTAL":
                            sourceDir = item.test_return_file.replace(
                                "marketname", market_name
                            )
                            m_name = sourceDir.split("/")[-1]
                            # shutil.copy2(
                            #     item.test_return_file,
                            #     f"{item.tDir}/R_{item.test_return_file.split('/')[-1]}",
                            # )
                            copy_tree(
                                sourceDir,
                                f"{item.tDir}/{m_name}/{market_name}",
                            )
                    # if index_result:
                    #     shutil.copy2(
                    #         item.test_index_file,
                    #         f"{item.tDir}/I_{item.test_index_file.split('/')[-1]}",
                    #     )
                    m_pass = False
            pd.DataFrame(data=selected_model, columns=self.column_name).to_csv(
                self.tDir + "/selected_model_results.csv"
            )

            # control searching a rule exploration
            if (len(selected_model) == 0) or (
                self.post_decision(selected_model) is None
            ):
                b_exit = False
                if (
                    soft_cond_retry == 1 or soft_cond_retry == 2 or soft_cond_retry == 3
                ):  # dynamic rule - trying multiple times according to the reducing parameters
                    retry = retry + 1
                else:  # static rules
                    retry = self.max_cnt + 1
            else:  # final decision
                if soft_cond_retry <= 3:  # for init and first best rules
                    print(" \nSelected base model by likely-hood: Disable")
                else:  # likely-hood for loose rules (e.g. second, third, forth best rules.. etc),
                    selected_model_list = np.array(selected_model)[
                        :, self.dict_col2idx["model_name"]
                    ].tolist()
                    selected_model_name = list(set(selected_model_list))
                    model_cnt = [
                        selected_model_list.count(selected_model_name[k_idx])
                        for k_idx in range(len(selected_model_name))
                    ]
                    pick_model = selected_model_name[int(np.argmax(model_cnt))]
                    selected_model = [
                        selected_model[i_idx]
                        for i_idx in range(len(selected_model))
                        if selected_model[i_idx][self.dict_col2idx["model_name"]]
                        == pick_model
                    ]
                    print(f" \nSelected base model by likely-hood: {pick_model}")

                idx = self.post_decision(selected_model)

                # copy files
                m_info = selected_model[idx]
                for market_name in list(RUNHEADER.mkidx_mkname.values()):
                    if market_name != "TOTAL":
                        sourceDir = m_info[
                            self.dict_col2idx["test_return_file"]
                        ].replace("marketname", market_name)
                        m_name = sourceDir.split("/")[-1]
                        copy_tree(
                            sourceDir,
                            f"{m_info[self.dict_col2idx['tDir']]}/final/{m_name}/{market_name}",
                        )
                if m_info[self.dict_col2idx["test_csv_file"]] is not None:
                    shutil.copy2(
                        m_info[self.dict_col2idx["test_csv_file"]],
                        f"{m_info[self.dict_col2idx['tDir']]}/final/{m_info[self.dict_col2idx['test_csv_file']].split('/')[-1]}",
                    )

                # shutil.copy2(
                #     m_info[self.dict_col2idx["test_return_file"]],
                #     f"{m_info[self.dict_col2idx['tDir']]}/final/R_{m_info[self.dict_col2idx['test_return_file']].split('/')[-1]}",
                # )
                # if index_result:
                #     shutil.copy2(
                #         m_info[self.dict_col2idx["test_index_file"]],
                #         f"{m_info[self.dict_col2idx['tDir']]}/final/I_{m_info[self.dict_col2idx['test_index_file']].split('/')[-1]}",
                #     )
                # if m_info[self.dict_col2idx["test_csv_file"]] is not None:
                #     shutil.copy2(
                #         m_info[self.dict_col2idx["test_csv_file"]],
                #         f"{m_info[self.dict_col2idx['tDir']]}/final/C_{m_info[self.dict_col2idx['test_csv_file']].split('/')[-1]}",
                #     )
                self.printout_model_info(m_info)

                b_exit = True

            if retry > self.max_cnt:
                if self.soft_cond:
                    self.select_criteria = float(self.select_criteria - 0.1)
                    soft_cond_retry = soft_cond_retry + 1
                    retry = 0
                    print(f"\nSoft condition is trying: {soft_cond_retry}")
                else:
                    b_exit = True

                if soft_cond_retry == 3:  # Diable 2,3,4,5 filter
                    ## Operation mode
                    # assert False, '[{}] Training more models, ' \
                    #               'there is no models passing stopping criteria'.format(dset_v)
                    # exit()

                    ## Test mode
                    print(
                        "\nPass data set - there is no models passing stopping criteria"
                    )
                    return 0
        return 1


def print_summary(data_list, th_dict):
    tmp = []
    for item in data_list:
        tmp.append(float(item.split("_")[17]))

    _str = f"f_result: {np.mean(np.array(tmp))}"
    for it in th_dict.items():
        _str = _str + " " + str(it[0]) + ":" + str(it[1]) + ", "
    fp = open("./save/result/selected/history.txt", "a")
    print(_str, file=fp)
    fp.close()
