import pandas as pd
import numpy as np


def get_consume_number(data, forward, ratio, key_name):
    data2 = data[forward:, 1:]
    data3 = data[:-forward, 1:]
    data2 = np.where(data2 - data3 > 0, 1, 0)  # up/down data

    data2 = data2[int(data2.shape[0] * ratio) : -1, :]

    for col in range(data2.shape[1]):
        nc_cnt = 0
        pc_cnt = 0
        nc_list = list()
        pc_list = list()
        for row in range(data2.shape[0]):
            if row > 0:
                if data2[row, col] == 0:
                    if data2[row, col] == data2[row - 1, col]:
                        nc_cnt = nc_cnt + 1
                    else:
                        if pc_cnt > 0:
                            pc_list.append(pc_cnt)
                            pc_cnt = 0
                else:
                    if data2[row, col] == data2[row - 1, col]:
                        pc_cnt = pc_cnt + 1
                    else:
                        if nc_cnt > 0:
                            nc_list.append(nc_cnt)
                            nc_cnt = 0
        print(
            "[{}] AVG. consume positive number:{}".format(
                key_name[col], np.mean(pc_list)
            )
        )
        print(
            "[{}] AVG. consume negative number:{}\n".format(
                key_name[col], np.mean(nc_list)
            )
        )


def get_call(data, forward, ratio, key_name):
    data2 = data[forward:, 1:]
    data3 = data[:-forward, 1:]
    data2 = np.where(data2 - data3 > 0, 1, 0)  # up/down data

    data2 = data2[int(data2.shape[0] * ratio) : -1, :]


if __name__ == "__main__":
    # Configuration
    forward = 20  # 1달 예측
    ratio = 1  # 최근 20년 분석
    ratio = 0.75  # 최근 5년 분석

    # Raw Data Load
    data = pd.read_csv("./datasets/rawdata/index_data/gold_index.csv")
    key_name = data.keys()
    data = data.values

    # 연속 일수 분석
    get_consume_number(data, forward, ratio, key_name)

    # 매수 타이밍 분석
    get_call(data, forward, ratio, key_name)
