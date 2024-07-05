import pandas as pd
import numpy as np


def DataPreprocessing():
    """
    该函数针对dermatology.data进行预处理
    """
    df = pd.read_csv('../../data/dermatology/dermatology.data')
    df = df[df.iloc[:, -2] != '?']
    last_column = df.columns[-1]
    df[last_column] = df[last_column].replace({6: 0})
    data = df.to_numpy()

    _data_x = data[:, 0:-1]
    _data_y = data[:, -1]
    data_x = _data_x.astype(np.float64)
    data_y = _data_y.astype(np.int32)
    min_vals = data_x[:, -1].min(axis=0)
    max_vals = data_x[:, -1].max(axis=0)
    range_vals = max_vals - min_vals
    data_x[:, -1] = (data_x[:, -1] - min_vals) / range_vals

    return data_x, data_y
