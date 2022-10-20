import re
import os
from datasets.windowing import (
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
    fun_mean,
    fun_cumsum,
    fun_cov,
    fun_cross_cov,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Correlation Plot
if __name__ == "__main__":
    X = "./datasets/rawdata/index_data/Synced_D_FilledData_new_080.csv"
    sample1, sample2 = 70, 750
    num_indices = 101
    
    for length in [sample1, sample2]:
        data = pd.read_csv(X)
        data = data.values[-length:, 1:num_indices]
        data_original = data

        for mv in [True, False]:
            data = data_original
            if mv == True:
                data = rolling_apply(fun_mean, data, 5)
            data = data[10:, :]

            for it in range(data.shape[1] - 1):
                _data = data[:, [it, it + 1]]
                _data = RobustScaler().fit_transform(_data)
                x, y, = (
                    _data[:, 0],
                    _data[:, 1],
                )
                plt_manager = plt.get_current_fig_manager()
                plt_manager.resize(1860, 980)
                plt.plot(x.tolist(), label="prediction (index)")
                plt.plot(y.tolist(), label="prediction (index)")

                if mv == True:
                    plt.savefig(
                        "./temp_dir/{}/mv/{}_{}.jpeg".format(
                            str(length), str(it), np.corrcoef(_data, rowvar=False)[0, 1]
                        ),
                        format="jpeg",
                        dpi=600,
                    )
                else:
                    plt.savefig(
                        "./temp_dir/{}/none_mv/{}_{}.jpeg".format(
                            str(length), str(it), np.corrcoef(_data, rowvar=False)[0, 1]
                        ),
                        format="jpeg",
                        dpi=600,
                    )
                plt.close()
