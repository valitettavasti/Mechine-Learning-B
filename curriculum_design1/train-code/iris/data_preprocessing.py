import pandas as pd


def DataPreprocessing():
    """
    该函数针对iris.data进行预处理
    """
    df = pd.read_csv('../../data/iris/iris.data')
    column = df.columns[-1]
    df[column] = df[column].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    data = df.to_numpy()

    data_x = data[:, 0:-1]
    data_y = data[:, -1]
    return data_x, data_y
