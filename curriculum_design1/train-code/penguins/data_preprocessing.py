import numpy as np
import pandas as pd


def DataPreprocessing():
    """
    该函数针对penguins_size.csv进行预处理
    """
    # 数据处理及去掉异常值
    df = pd.read_csv('../../data/penguins/penguins_size.csv')
    df = df.dropna()
    df = df[df['sex'] != '.']

    # 将非数据类型转化为数据类型
    df['species'] = df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df['island'] = df['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
    df['sex'] = df['sex'].replace({'FEMALE': -1, 'MALE': 1})

    # 将数组转化为numpy数组
    data = df.to_numpy()
    data_x = data[:, 1:]
    data_y = data[:, 0]

    min_vals = data_x[:, 1:-1].min(axis=0)
    max_vals = data_x[:, 1:-1].max(axis=0)
    range_vals = max_vals - min_vals
    data_x[:, 1:-1] = (data_x[:, 1:-1] - min_vals) / range_vals

    return data_x, data_y
