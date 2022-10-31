import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Bypass the need to install Tkinter GUI framework

import matplotlib.pyplot as plt

data = pd.read_csv("./temp/expected_returns_std.csv")

std = data["expected_return_std"]

m_min = np.min(std)
m_max = np.max(std)
bins = np.arange(m_min, m_max, 1)

y_data = list()
x_data = list()
for idx in range(len(bins)):
    if idx == 0:
        pass
    else:
        where_rows = (bins[idx - 1] < data["expected_return_std"]) & (
            data["expected_return_std"] < bins[idx]
        )
        y_data.append(np.sum(data[where_rows]["expected_return"]))
        x_data.append(bins[idx])

plt.plot(x_data, y_data)
plt.savefig("./temp/test.png")
plt.close()
