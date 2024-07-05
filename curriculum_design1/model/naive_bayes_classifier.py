import numpy as np


# 计算给定测试数据和标签的多项式概率密度函数
def calculate_pdf(test_data, tag_mean, tag_std, p_tag):
    test_tag = np.zeros(test_data.shape)
    for i in range(test_data.shape[1]):
        test_tag[:, i] = 1 / (np.sqrt(2 * np.pi) * tag_std[i]) * np.exp(
            -(test_data[:, i] - tag_mean[i]) ** 2 / (2 * tag_std[i] ** 2))
    p_test_tag = np.ones(test_tag.shape[0])
    p_test_tag *= p_tag
    for i in range(test_data.shape[1]):
        p_test_tag *= test_tag[:, i]
    return p_test_tag


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器
    tags_data[]:每种类别的测试数据列表
    p_tags[]:每种类别的先验概率列表
    tags_mean[]:每种类别的均值列表
    tags_std[]:每种类别的标准差列表
    elements:类别标签的唯一值列表
    """

    def __init__(self) -> None:
        self.tags_data = []
        self.p_tags = []
        self.tags_mean = []
        self.tags_std = []
        self.elements = []

    # 训练模型方法
    def train(self, train_x, train_y):
        self.elements = np.unique(train_y)
        for element in self.elements:
            self.tags_data.append(train_x[train_y == element])
        for tag in self.tags_data:
            self.p_tags.append(tag.shape[0] / train_x.shape[0])
            self.tags_mean.append(np.mean(tag, axis=0))
            tag_std = np.nanstd(tag, axis=0)
            tag_std[tag_std == 0] = 1e-6
            self.tags_std.append(tag_std)

    # 预测函数
    def predict(self, test_x, test_y):
        p_test_tags = []
        for i in range(len(self.elements)):
            p_test_tags.append(calculate_pdf(
                test_x, self.tags_mean[i], self.tags_std[i], self.p_tags[i]
            ))
        # 将每个数据点的所有类别概率合并为一个数组
        concatenated_array_col = None
        for tag in p_test_tags:
            tags = tag.reshape(1, -1)
            if concatenated_array_col is None:
                concatenated_array_col = tags.T
            else:
                concatenated_array_col = np.hstack((concatenated_array_col, tags.T))
        index_max = np.argmax(concatenated_array_col, axis=1)
        acc = np.sum(index_max == test_y)
        print('Naive bayes classifier test Accuracy', acc / test_y.shape[0])
        return index_max


# 朴素贝叶斯的函数形式调用方法
def naive_bayes_classifier_function(train_x, train_y, test_x, test_y):
    model = NaiveBayesClassifier()
    model.train(train_x, train_y)
    model.predict(test_x, test_y)
