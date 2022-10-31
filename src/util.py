# -*- coding: utf-8 -*-
"""
@author: kim KyungMin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.envs.registration import registry
from contextlib import contextmanager
import time
import collections
import numpy as np
import pickle
import pandas as pd
import os
import re

# from sklearn.externals import joblib
import joblib
import cloudpickle
import json
import datetime

# from memory_profiler import profile
from operator import itemgetter
from itertools import groupby
import sklearn.metrics as metrics
import collections
import sys


def get_domain_on_CDSW_env(domain):
    for it in ["cdsw_20", "cdsw_60", "cdsw_120"]:
        if it in domain:
            fp = open("{}.txt".format(domain), "r")
            domain = fp.readline().replace("\n", "")
            fp.close()
    return domain


def check_training_status(b_activate, count, target_name, forward_ndx):
    r_dir = "./"
    model_dir = "./save/model/rllearn/"
    _status = None
    cnt = 0
    min_num_models = 100

    if b_activate:
        files = [
            it
            for it in os.listdir(model_dir)
            if (target_name in it) and ("T" + forward_ndx) in it
        ]

        for file in files:
            try:
                r2 = re.compile(".*pkl")
                if (
                    len(list(filter(r2.match, os.listdir(model_dir + file))))
                    >= min_num_models
                ):
                    cnt = cnt + 1
            except FileNotFoundError:
                pass
        _status = True if cnt >= count else False
    else:
        _status = True
    return _status, cnt


def f_error_test(candidate_model, selected_model, performence_stacks):
    pd.set_option("mode.chained_assignment", None)
    CANDIDATE = 0
    SELECT = 1
    if selected_model is None:
        selected_model = candidate_model

    error_test = list()
    for file_name in [candidate_model[3], selected_model[3]]:
        file_name = "{}/validation/{}".format(
            file_name,
            [
                file
                for file in os.listdir("{}/validation".format(file_name))
                if file.endswith(".csv")
            ].pop(),
        )
        model_result_data = pd.read_csv(file_name, index_col=0)
        model_result_data_recent = model_result_data.iloc[
            -10:
        ].copy()  # recent 10 working days

        metrics_mae = (
            np.square(
                model_result_data_recent["P_return"]
                - model_result_data_recent["Return"]
            )
        ).mean(axis=0)
        metrics_ratio = np.sum(model_result_data["P_20days"]) / len(
            model_result_data["P_20days"]
        )
        metrics_accuray = metrics.accuracy_score(
            model_result_data["20days"], model_result_data["P_20days"]
        )
        if metrics_ratio > 0.5:
            metrics_ratio = 1 - metrics_ratio

        # if len(error_test) == 0:
        #     performence_stacks.append([candidate_model[0], candidate_model[1],
        #     candidate_model[2], candidate_model[3], metrics_mae, metrics_ratio, metrics_accuray])
        performence_stacks.append(
            [
                candidate_model[0],
                candidate_model[1],
                candidate_model[2],
                candidate_model[3],
                metrics_mae,
                metrics_ratio,
                metrics_accuray,
            ]
        )
        error_test.append(metrics_mae)

    selected_model = (
        candidate_model
        if error_test[CANDIDATE] <= error_test[SELECT]
        else selected_model
    )
    pd.set_option("mode.chained_assignment", "warn")

    return selected_model, performence_stacks


def get_unique_list(var_list):
    return list(map(itemgetter(0), groupby(var_list)))


def find_date(source_dates, str_date, search_interval):
    max_test = 1
    datetime_obj = datetime.datetime.strptime(str_date, "%Y-%m-%d")
    while True:
        return_date = np.argwhere(source_dates == datetime_obj.strftime("%Y-%m-%d"))
        if len(return_date) > 0:
            return return_date[0][0]
        else:
            datetime_obj += datetime.timedelta(days=search_interval)
            max_test = max_test + 1
        assert (
            max_test <= 10
        ), "check your date object, might be something wrong (Required Format: yyyy-mm-dd )!!!"


def _replace_cond(f_cond, matrix):
    if matrix.ndim == 2:
        matrix = np.append([np.zeros(matrix.shape[1])], matrix, axis=0)
        matrix = np.array(matrix, dtype=np.float32)
        tmp_idx = np.argwhere(f_cond(matrix))
        while len(tmp_idx) > 0:
            for row, col in tmp_idx:
                matrix[row, col] = matrix[row - 1, col]
            tmp_idx = np.argwhere(f_cond(matrix))
        return matrix[1:, :]
    else:  # 'Not defied yet'
        return matrix  # Assume already pre-processed


def _remove_cond(f_cond, matrix, target_col=None, axis=0, det=True):
    if target_col is None:
        t = np.argwhere(f_cond(matrix)[:, axis] == det).T.squeeze().tolist()
    else:
        t = np.argwhere(f_cond(matrix[:, target_col]) == det).T.squeeze().tolist()

    if len(t) > 0:
        matrix = np.delete(matrix, t, axis=axis)
    assert matrix.ndim == 2, "2D matrix only for now"

    return matrix, t


def current_y_unit(target_name):
    if target_name in [
        "US10YT",
        "GB10YT",
        "DE10YT",
        "KR10YT",
        "CN10YT",
        "JP10YT",
        "BR10YT",
    ]:
        return "percent"
    else:
        return "prc"


def current_x_unit(d_f_summary, target_name):
    if "-" in target_name:
        target_name = target_name.split("-")[0]

    selected_item = d_f_summary["var_name"]
    T1 = selected_item == target_name
    if target_name in vol_index:
        return "volatility"
    if d_f_summary[T1]["units"].values[0] == "%":
        return "percent"
    if d_f_summary[T1]["description"].values[0].upper() == "SWAP":
        return "percent"
    if d_f_summary[T1]["description"].values[0].upper() == "LIBOR":
        return "percent"
    if d_f_summary[T1]["description"].values[0].upper() == "REPO":
        return "percent"
    if d_f_summary[T1]["description"].values[0].upper() == "SHIBOR":
        return "percent"
    if d_f_summary[T1]["description"].values[0].upper() == "TAIBOR":
        return "percent"
    return "prc"


def _ordinary_return_prc(matrix=None, v_init=None, v_final=None):
    np.seterr(divide="ignore", invalid="ignore")
    if matrix is not None:  # 2D case daily return
        diff_data = np.diff(matrix, axis=0)
        _x = np.abs(matrix[:-1, :])
        x = np.where(_x == 0, np.inf, _x)
        o_return = np.divide(diff_data, x) * 100
        # o_return = (diff_data / np.abs(matrix[:-1, :])) * 100
        assert np.allclose(o_return.shape, diff_data.shape), "dimension error"
        assert matrix.ndim == 2, "dimension error"
        o_return = np.append([np.zeros(o_return.shape[1])], o_return, axis=0)
    else:  # 1D or 2D case ordinary return
        assert (v_init is not None) and (v_final is not None), "empty"
        assert v_init.shape == v_final.shape, "dimension error"
        _x = np.abs(v_init)
        x = np.where(_x == 0, np.inf, _x)
        o_return = np.divide((v_final - v_init), x) * 100
        # o_return = ((v_final - v_init) / np.abs(v_init)) * 100
    np.seterr(divide="warn", invalid="warn")
    return o_return


def _ordinary_return_percent(matrix=None, v_init=None, v_final=None):
    np.seterr(divide="ignore", invalid="ignore")
    if matrix is not None:  # 2D case daily return
        o_return = np.diff(matrix, axis=0)
        assert matrix.ndim == 2, "dimension error"
        o_return = np.append([np.zeros(o_return.shape[1])], o_return, axis=0)
    else:  # 1D or 2D case ordinary return
        assert (v_init is not None) and (v_final is not None), "empty"
        assert v_init.shape == v_final.shape, "dimension error"
        o_return = v_final - v_init
    np.seterr(divide="warn", invalid="warn")
    return o_return


def ordinary_return(matrix=None, v_init=None, v_final=None, unit="prc"):
    if unit == "prc":
        o_return = _ordinary_return_prc(matrix, v_init, v_final)
    elif (unit == "percent") or (unit == "volatility"):
        o_return = _ordinary_return_percent(matrix, v_init, v_final)
    else:
        o_return = None
    return _replace_cond(np.isinf, o_return)


def dict2json(file_name, _dict):
    # Save to json
    with open(file_name, "w") as f_out:
        json.dump(_dict, f_out)
    f_out.close()


def json2dict(file_name):
    with open(file_name, "r") as f_out:
        _dict = json.load(f_out)
    f_out.close()
    return _dict


@contextmanager
def restore_agents_status(self):
    self.env.env_method("clear_cache_test_step")  # clear l2_cache for test step
    tmp_mode = self.env.get_attr("mode")
    tmp_current_step = self.env.get_attr("current_step")
    tmp_eof = self.env.get_attr("eof")
    tmp_steps_beyond_done = self.env.get_attr("steps_beyond_done")
    yield
    self.env.set_attr("mode", tmp_mode)
    self.env.set_attr("current_step", tmp_current_step)
    self.env.set_attr("eof", tmp_eof)
    self.env.set_attr("steps_beyond_done", tmp_steps_beyond_done)


def discount_reward(rewards, dones, target):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    c_cnt = list()
    ret = 0
    consequtive_cnt = 0

    f_gamma = lambda _consequtive_cnt, _target: 1 - (
        1 / (1 + np.exp(-_consequtive_cnt + _target))
    )

    for done in dones[::-1]:
        if not done:
            consequtive_cnt = consequtive_cnt + 1
        else:
            consequtive_cnt = 0
        c_cnt.append(consequtive_cnt)
    c_cnt = c_cnt[::-1]

    for idx in range(len(c_cnt)):
        if not dones[idx]:
            for num in range(c_cnt[idx]):
                if not num:
                    gamma = 1
                else:
                    gamma = f_gamma(num, target)
                ret = ret + gamma * rewards[idx] * (1.0 - dones[idx])
                idx = idx + 1
        else:
            ret = rewards[idx]
        discounted.append(ret)
        ret = 0

    return discounted


def consecutive_junk(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def consecutive_true(data):
    cnt = 0
    max_cnt = 0
    for i in range(len(data)):
        if not data[i]:
            cnt = cnt + 1
        else:
            cnt = 0
        max_cnt = np.max([cnt, max_cnt])
    return max_cnt


def make(_id):
    try:
        registry.env_specs[_id]
    except KeyError:
        # Force to load __init__.py
        import __init__
    return gym.make(id=_id)


# time check
@contextmanager
def funTime(func_name):
    start = time.clock()
    yield
    end = time.clock()
    interval = end - start
    print("\n== Time cost for [{0}] : {1}".format(func_name, interval))


# if an object contains one more char type then return False
def charIsNum(string):
    bln = True
    for ch in string:
        if ord(ch) > 48 | ord(ch) < 57:
            bln = False
    return bln


def is_empty(a):
    return not a and isinstance(a, collections.Iterable)


def nanTozero(result):
    return np.where(np.isnan(result), 0, result)


# Get file names in directory
def getFileList(file_directory):
    file_names = []
    for file_list in os.listdir(file_directory):
        file_names.append("{0}/{1}".format(file_directory, file_list))

    return file_names


# Serialization with pickle [0: pickle, 1: joblib, 2: cloudpickle]
def writeFile(file_name, data, pickle_type=0):
    file_name = "{0}.pkl".format(file_name)
    if pickle_type == 1:
        joblib.dump(data, file_name)
    else:
        with open(file_name, "wb") as fp:
            if pickle_type == 0:
                pickle.dump(data, fp, protocol=4)
            elif pickle_type == 2:
                cloudpickle.dump(data, fp)
            fp.close()
    print("\nExporting files {0}.... Done!!!".format(file_name))


def read_pickle(file_name):
    fp = open(file_name, "rb")
    data = pickle.load(fp)
    fp.close()
    return data


def write_pickle(data, file_name):
    fp = open(file_name, "wb")
    data = pickle.dump(data, fp)
    fp.close()


def str_join(sep, *args):
    return "{}".format(sep).join(args)


def remove_duplicaated_dict_in_list(m_list):
    return list(
        map(
            dict,
            collections.OrderedDict.fromkeys(
                tuple(sorted(it.items())) for it in m_list
            ),
        )
    )


# Deserialization with pickle
# @profile
def loadFile(file_name, pickle_type=0):
    data = None

    # Load file
    file_name = "{0}.pkl".format(file_name)
    if pickle_type == 1:
        data = joblib.load(file_name)
    else:
        with open(file_name, "rb") as fp:
            if pickle_type == 0:
                data = pickle.load(fp)
            elif pickle_type == 2:
                data = cloudpickle.load(fp)
            fp.close()
    print("\nLoading files {0}.... Done!!!".format(file_name))

    return data


def get_manual_vars_additional():
    return loadFile("./c_vars")


def _trans_val(data, unit, t_unit="defferential"):
    if t_unit == "defferential":
        if unit == "prc":
            diff_data = np.diff(data)
            _x = np.abs(data[:-1])
            x = np.where(_x == 0, np.inf, _x)
            return np.append([np.zeros(1)], np.divide(diff_data, x) * 100)
        elif (unit == "percent") or (unit == "volatility"):
            return data
        else:
            return data
    else:
        assert False, "Not Defined yet!!!"


def trans_val(
    x_data=None,
    y_data=None,
    x_index=None,
    t_unit="defferential",
    f_desc=None,
    target_name=None,
):
    X_val, Y_val, X_unit, Y_unit = None, None, None, None
    d_f_summary = pd.read_csv(f_desc)

    if y_data is not None:
        Y_val, Y_unit = np.zeros(y_data.shape), list()
        unit = current_y_unit(target_name)
        Y_val = _trans_val(y_data, unit, t_unit)
        Y_unit.append(unit)

    if x_data is not None:
        X_val, X_unit = np.zeros(x_data.shape), list()
        for idx in range(x_data.shape[1]):
            unit = current_x_unit(d_f_summary, x_index[idx])
            X_val[:, idx] = _trans_val(x_data[:, idx], unit, t_unit)
            X_unit.append(unit)

    return X_val, Y_val, X_unit, Y_unit


# Serialization with DataFrame
def pdToFile(file_name, df):
    file_name = "{0}.csv".format(file_name)
    df.to_csv(file_name, index=False)
    print("\nExporting files {0}.... Done!!!".format(file_name))


# Deserialization with DataFrame
def fileToPd(file_name):
    # Load file
    file_name = "{0}.csv".format(file_name)
    data = pd.read_csv(file_name, index_col=False)
    print("\nLoading files {0}.... Done!!!".format(file_name))
    return data


def npToFile(file_name, X, format="%s"):
    file_name = "{0}.csv".format(file_name)
    np.savetxt(file_name, X, fmt=format, delimiter=",")


def print_flush(item):
    sys.stdout.write("\r>> " + item)
    sys.stdout.flush()


def inline_print(item, cnt, interval):
    if cnt > 0:
        if (cnt % interval) == 0:
            print(item, sep=" ", end="", flush=True)


def configure_learning_rate(num_samples_per_epoch, global_step, FLAGS, tf):
    """Configures the learning rate.

      Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """

    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch FLAGS.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(
        num_samples_per_epoch * FLAGS.num_epochs_per_decay / FLAGS.batch_size
    )

    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == "exponential":
        return tf.compat.v1.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True,
            name="exponential_decay_learning_rate",
        )
    elif FLAGS.learning_rate_decay_type == "fixed":
        return tf.constant(FLAGS.learning_rate, name="fixed_learning_rate")
    elif FLAGS.learning_rate_decay_type == "polynomial":
        return tf.compat.v1.train.polynomial_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.end_learning_rate,
            power=1.0,
            cycle=False,
            name="polynomial_decay_learning_rate",
        )
    else:
        raise ValueError(
            "learning_rate_decay_type [%s] was not recognized"
            % FLAGS.learning_rate_decay_type
        )


def configure_optimizer(learning_rate, FLAGS, tf):
    """Configures the optimizer used for training.

    Args:
    learning_rate: A scalar or `Tensor` learning rate.

    Returns:
    An instance of an optimizer.

    Raises:
    ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == "adadelta":
        optimizer = tf.compat.v1.train.AdadeltaOptimizer(
            learning_rate, rho=FLAGS.adadelta_rho, epsilon=FLAGS.opt_epsilon
        )
    elif FLAGS.optimizer == "adagrad":
        optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value,
        )
    elif FLAGS.optimizer == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon,
        )
    elif FLAGS.optimizer == "ftrl":
        optimizer = tf.compat.v1.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2,
        )
    elif FLAGS.optimizer == "momentum":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate, momentum=FLAGS.momentum, name="Momentum"
        )
    elif FLAGS.optimizer == "rmsprop":
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon,
        )
    elif FLAGS.optimizer == "sgd":
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("Optimizer [%s] was not recognized" % FLAGS.optimizer)

    return optimizer


def get_init_fn(FLAGS, tf, slim):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
    An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.compat.v1.logging.info(
            "Ignoring --checkpoint_path because a checkpoint already exists in %s"
            % FLAGS.train_dir
        )
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(",")
        ]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
            else:
                variables_to_restore.append(var)

    if tf.io.gfile.isdir(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.compat.v1.logging.info("Fine-tuning from %s" % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars,
    )


def get_variables_to_train(FLAGS, tf):
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.compat.v1.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(",")]

    variables_to_train = []
    for scope in scopes:
        variables = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope
        )
        variables_to_train.extend(variables)
    return variables_to_train


def get_common_variables(data_ids_names, desc):
    m_vars = list()
    for ids_name in data_ids_names:
        cond = desc["var_name"] == ids_name

        if desc[cond]["category"].tolist()[0] == "Market Index":
            None

        for var_desc in list(set(vars_desc["category"])):
            None

    desc = pd.read_csv(RUNHEADER.var_desc)
    categories = list(desc["category"])
    quantise = list()
    for it in list(set(desc["category"])):
        quantise.append([it, int(categories.count(it) * 0.2)])

    num_max_vars = OrderedDict(quantise)
    new_ids_to_var_names = list()
    duplicate_idx = list()
    for key, max_val in num_max_vars.items():
        cnt = 0
        for ids, ids_name in ids_to_var_names.items():
            if duplicate_idx.count(ids) == 0:
                if "-" in ids_name:
                    new_ids_to_var_names.append([int(ids), ids_name])
                    duplicate_idx.append(ids)
                else:
                    t_key = desc[desc["var_name"] == ids_name]["category"].tolist()[0]
                    if (cnt <= max_val) and (key == t_key):
                        new_ids_to_var_names.append([int(ids), ids_name])
                        cnt = cnt + 1
                        duplicate_idx.append(ids)
    new_ids_to_var_names = sorted(new_ids_to_var_names, key=lambda aa: aa[0])
    selected_idxs = np.array(new_ids_to_var_names, dtype=np.object)[:, 0].tolist()

    # update
    data = data[:, selected_idxs]
    ids_to_var_names = OrderedDict(new_ids_to_var_names)


vol_index = [
    "BPVIX__unvsl_clos_prc",
    "CESIAPAC__unvsl_clos_prc",
    "CESIAUD__unvsl_clos_prc",
    "CESICAD__unvsl_clos_prc",
    "CESICHF__unvsl_clos_prc",
    "CESICMEA__unvsl_clos_prc",
    "CESICNY__unvsl_clos_prc",
    "CESIEM__unvsl_clos_prc",
    "CESIEUR__unvsl_clos_prc",
    "CESIG10__unvsl_clos_prc",
    "CESIGBP__unvsl_clos_prc",
    "CESIJPY__unvsl_clos_prc",
    "CESILTAM__unvsl_clos_prc",
    "CESINOK__unvsl_clos_prc",
    "CESINZD__unvsl_clos_prc",
    "CESISEK__unvsl_clos_prc",
    "CESIUSD__unvsl_clos_prc",
    "EUVIX__unvsl_clos_prc",
    "JYVIX__unvsl_clos_prc",
    "V6I1__unvsl_clos_prc",
    "V6I4__unvsl_clos_prc",
    "V6I5__unvsl_clos_prc",
    "V6I6__unvsl_clos_prc",
    "V6I7__unvsl_clos_prc",
    "V6I8__unvsl_clos_prc",
    "VHSI__unvsl_clos_prc",
    "VIXMO__unvsl_clos_prc",
    "VOL__unvsl_clos_prc",
    "VST1ME__unvsl_clos_prc",
    "VXN__unvsl_clos_prc",
    "VXTLT__unvsl_clos_prc",
]
