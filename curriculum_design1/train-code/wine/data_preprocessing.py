import pandas as pd


def DataPreprocessing():
    """
    该函数针对wine.data进行预处理
    """
    df = pd.read_csv('../../data/wine/wine.data')
    last_column = df.columns[0]
    df[last_column] = df[last_column].replace({1: 0, 2: 1, 3: 2})

    data = df.to_numpy()
    data_x = data[:, 1:]
    data_y = data[:, 0]
    min_vals = data_x.min(axis=0)
    max_vals = data_x.max(axis=0)
    range_vals = max_vals - min_vals
    data_x = (data_x - min_vals) / range_vals
    return data_x, data_y
