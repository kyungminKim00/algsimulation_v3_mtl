import numpy as np
from scipy.stats import spearmanr
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


if __name__ == "__main__":
    ids_to_class_names = load_file(
        "./datasets/rawdata/fund_data/348_First_Try/fund_id.pkl", "pkl"
    )
    fund_data = load_file("./datasets/rawdata/fund_data/348_First_Try/tri.npy", "npy")

    # delete 1 index funds
    tmp_list = list()
    for item in list(ids_to_class_names.items()):
        tmp = list(item)
        if tmp[0] != 1:
            tmp_list.append(tmp)
    ids_to_class_names = dict(tmp_list)
    fund_data = np.delete(fund_data, 1, axis=1)
    assert len(ids_to_class_names) == fund_data.shape[-1], "shapes are different"

    # calculate correlation
    _fund_data = MinMaxScaler().fit_transform(fund_data)
    cor, p = spearmanr(_fund_data, axis=0)

    n_row, n_col = cor.shape
    data = cor

    data[0] = -np.inf
    data[:, 0] = -np.inf

    # UT matrix init
    for i in range(n_col):
        for j in range(n_row):
            if (j > i) or (j == i):
                data[j, i] = -np.inf

    # remove duplicated variables
    for j in range(n_row):
        tmp = data[j, :]
        idx = np.squeeze(np.argwhere(tmp > 0.95))  # default 144
        # idx = np.squeeze(np.argwhere(tmp > 0.80))  # default 30
        data[idx, :] = -np.inf
        data[:, idx] = -np.inf

    # variable selection
    vs = list()
    for j in range(n_row):
        tmp = data[j, :]
        if (len(np.argwhere(tmp > -np.inf)) > 0) or (
            j == 0
        ):  # force to insert cash action
            vs.append(j)
    print("[{}] selected variables: {}".format(len(vs), vs))
    fund_data = fund_data[:, vs]
    # variable selection - adjust dictionary
    tmp_list = list()
    k = 0
    insert_idx = 0
    for item in list(ids_to_class_names.items()):
        tmp = list(item)
        if k == 0:  # insert cash actions Todo: caution Remove later with proper dataset
            tmp_list.append([insert_idx, ids_to_class_names[k]])
            insert_idx = insert_idx + 1
        else:
            if k in vs:
                tmp_list.append([insert_idx, tmp[1]])
                insert_idx = insert_idx + 1
        k = k + 1

    ids_to_class_names = dict(tmp_list)
    assert len(ids_to_class_names) == fund_data.shape[-1], "shapes are different"

    """File save
    """
    fund__dates = load_file("./datasets/rawdata/fund_data/tri_dates.npy", "npy")

    # to binary file for convert_fs_v2.py
    write_file("./datasets/rawdata/fund_data/fund_id.pkl", ids_to_class_names, "pkl")
    write_file("./datasets/rawdata/fund_data/tri.npy", fund_data, "npy")

    # to csv file for convert_fs_v4.py
    to_csv = np.insert(np.array(fund_data, dtype=np.object), 0, fund__dates, axis=1)
    to_csv = np.insert(
        to_csv, 0, ["TradeDate"] + list(ids_to_class_names.values()), axis=0
    )
    pd.DataFrame(to_csv).to_csv(
        "./datasets/rawdata/fund_data/fund_data.csv", header=None, index=None
    )

    print("Done!!")
