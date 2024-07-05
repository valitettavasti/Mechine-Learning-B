import pandas as pd


def DataPreprocessing():
    """
    该函数针对ionosphere.data进行预处理
    """
    df = pd.read_csv('../../data/ionosphere/ionosphere.data')
    last_column = df.columns[-1]
    df[last_column] = df[last_column].replace({'b': 0, 'g': 1})
    last_column = df.columns[-1]
    df[last_column] = df[last_column].replace({6: 0})

    data = df.to_numpy()
    data_x = data[:, 0:-1]
    data_y = data[:, -1]
    return data_x, data_y
