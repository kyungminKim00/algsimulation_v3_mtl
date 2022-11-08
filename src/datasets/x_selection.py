import copy
import pickle
from collections import OrderedDict

import bottleneck as bn
import header.index_forecasting.RUNHEADER as RUNHEADER
import joblib
import numpy as np
import pandas as pd
from ray.util.joblib import register_ray
from scipy.stats import spearmanr
from util import find_date

from datasets.windowing import fun_mean, rolling_apply

register_ray()


def load_file(file_location, file_format):
    with open(file_location, "rb") as fp:
        if file_format == "npy":
            return np.load(fp)
        elif file_format == "pkl":
            return pickle.load(fp)
        else:
            raise ValueError("non-support file format")


def write_file(file_location, data, file_format):
    if file_format == "npy":
        np.save(file_location, data)

    if file_format == "pkl":
        with open(file_location, "wb") as fp:
            pickle.dump(data, fp)


def rm_indx(data, idx):
    data = np.delete(data, idx, axis=1)
    return data


def get_uniqueness(
    file_name=None,
    target_name=None,
    from_file=True,
    _data=None,
    _dict=None,
    opt=None,
    th=0.94,
    eod=None,
):
    th: float = float(th)
    if not from_file:
        assert _dict is not None, "variable name should be given"

    if from_file:
        sd_data: pd.DataFrame = pd.read_csv(file_name)
        if eod is not None:
            assert False, "안쓰이는 걸로 생각 하고 있음. 로직 맞는지 생각 해 보기"
            _dates = sd_data.values
            e_test_idx = (
                find_date(_dates, eod, -1)
                if len(np.argwhere(_dates == eod)) == 0
                else np.argwhere(_dates == eod)[0][0]
            )
            sd_data = sd_data.iloc[e_test_idx - 70 : e_test_idx, :]
        col_name: pd.Index = sd_data.columns
        dates: np.ndarray = np.array(sd_data["TradeDate"])
        sd_data: np.ndarray = sd_data.values[:, 1:]

    else:
        assert isinstance(_dict, dict), "type error"
        c_vars = list(_dict.values())

        col_name = ["TradeDate"] + c_vars
        dates = np.array(_data[:, 0])
        sd_data = _data[:, 1:]
    original_num: int = sd_data.shape[1]

    # target  data
    original_dates = dates
    original_sd_data = sd_data

    dates = dates[: RUNHEADER.m_pool_sample_end]
    sd_data = sd_data[: RUNHEADER.m_pool_sample_end, :]

    if opt == "mva":
        sd_data = sd_data.astype(float)
        sd_data = bn.move_mean(sd_data, window=5, min_count=1, axis=0)

    # # remove data
    # sd_data = rm_indx(sd_data, [17, 49])
    # write_file('./datasets/rawdata/fund_data/sdata_new.npy', sd_data, 'npy')

    # cor, p = spearmanr(sd_data, axis=0)
    # n_row, n_col = cor.shape
    # data = cor

    with joblib.parallel_backend("ray"):
        cor, _ = spearmanr(sd_data, axis=0)
    n_row, n_col = cor.shape
    data = cor

    # UT matrix init
    for i in range(n_col):
        for j in range(n_row):
            if j > i:
                data[j, i] = -np.inf
            if j == i:
                data[j, i] = th

    # remove duplicated variables
    for j in range(n_row):
        tmp = data[j, :]
        idx = np.squeeze(np.argwhere(tmp > th))
        data[idx, :] = -np.inf
        data[:, idx] = -np.inf

    # variable selection
    vs = list()
    for j in range(n_row):
        tmp = data[j, :]
        if len(np.argwhere(tmp > -np.inf)) > 0:
            vs.append(j)
    print(f"\n selected variables: {len(vs)} from {original_num}")
    sd_data = original_sd_data[:, vs]

    """File save
    """
    # write_file('./datasets/rawdata/fund_data/sdata_new.npy', sd_data, 'npy')

    if from_file:
        col_name = [col_name[0]] + [col_name[idx + 1] for idx in vs]
        dates = np.reshape(original_dates, [original_dates.shape[0], 1])
        sd_data = np.append(dates, sd_data, axis=1)
        pd.DataFrame(sd_data, columns=col_name).to_csv(target_name, index=None)
        print("Done!!")
    else:
        col_name = [col_name[idx + 1] for idx in vs]
        return sd_data, OrderedDict(zip(list(np.arange(len(col_name))), col_name))


def get_uniqueness_without_dates(
    file_name=None,
    target_name=None,
    from_file=True,
    _data=None,
    _dict=None,
    opt=None,
    th=0.975,
):
    assert (
        (from_file is False) or (_data is not None) or (file_name is None)
    ), "None Defined method"

    # col_name = _dict if type(_dict) is list else list(_dict.values())
    col_name = _dict if isinstance(_dict, list) else list(_dict.values())

    th = float(th)
    if not from_file:
        assert _dict is not None, "variable name should be given"

    # sd_data = _data
    # original_sd_data = _data

    sd_data = copy.deepcopy(_data)
    original_sd_data = copy.deepcopy(_data)

    if opt == "mva":
        sd_data = bn.move_mean(sd_data.astype(float), window=5, min_count=1, axis=0)

    # cor, p = spearmanr(sd_data, axis=0)
    # n_row, n_col = cor.shape
    # data = cor

    with joblib.parallel_backend("ray"):
        cor, _ = spearmanr(sd_data, axis=0)
    n_row, n_col = cor.shape
    data = cor

    # UT matrix init
    for i in range(n_col):
        for j in range(n_row):
            if j > i:
                data[j, i] = -np.inf
            if j == i:
                data[j, i] = th

    # remove duplicated variables
    for j in range(n_row):
        tmp = data[j, :]
        idx = np.squeeze(np.argwhere(tmp > th))
        data[idx, :] = -np.inf
        data[:, idx] = -np.inf

    # variable selection
    vs = list()
    for j in range(n_row):
        tmp = data[j, :]
        if len(np.argwhere(tmp > -np.inf)) > 0:
            vs.append(j)

    # modifying
    sd_data = None
    sd_data = original_sd_data[:, vs]
    col_name = [col_name[idx] for idx in vs]

    return sd_data, OrderedDict(zip(list(np.arange(len(col_name))), col_name))


# if __name__ == '__main__':
# #     # sd_data = load_file('./datasets/rawdata/fund_data/sdata.npy', 'npy')
# #     # file_name = './datasets/rawdata/index_data/merged_data.csv'
# #     # target_name = './datasets/rawdata/index_data/merged_data_new.csv'
# #
#     # Synced_D_FilledData_xx용 으로 만 활용
#     file_name = '../datasets/rawdata/index_data/Synced_D_FilledData.csv'
#     target_name = '../datasets/rawdata/index_data/Synced_D_FilledData_new_08.csv'
#     get_uniqueness(file_name=file_name, target_name=target_name, from_file=True, _data=None, _dict=None, th=0.80)
