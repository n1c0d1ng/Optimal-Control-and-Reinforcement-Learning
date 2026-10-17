# Systemidentifikation

import numpy as np

def read_data(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=0)
    time_grid = raw[:, 0]           # (N,)
    data = raw[:, 1:]               # (N, M) – M Trajektorien
    return time_grid, data,raw

[time_grid,data,raw] = read_data(path="logistic_data.csv")
print(data.shape)

