from sklearn import tree
import numpy as np
from scipy import stats


class RandomForest:
    """
    随机森林模型(以决策树为基学习器)
    model_all:存储所有决策树模型的列表
    y_pred_all:存储所有模型预测结果的列表
    accuracy_all:存储所有模型的准确率的列表
    base_learner_times:基学习器（决策树）的数量(默认为10)
    """

    def __init__(self, base_learner_times=10):
        self.model_all = []
        self.y_pred_all = []
        self.accuracy_all = []
        self.base_learner_times = base_learner_times

    def train(self, train_x, train_y):
        for i in range(1, self.base_learner_times + 1):
            choice_array = np.random.choice(train_x.shape[0], train_x.shape[0], replace=True)
            train_x_choice = train_x[choice_array, :]
            train_y_choice = train_y[choice_array]
            model = tree.DecisionTreeClassifier(
                random_state=1,
                criterion='entropy',
                max_features=int(train_x.shape[1] / 3))  # 最大特征数为三分之一的特征总数
            model.fit(train_x_choice, train_y_choice)
            self.model_all.append(model)

    def predict(self, test_x, test_y):
        for i in range(len(self.model_all)):
            y_pred = self.model_all[i].predict(test_x)
            y_pred = y_pred.reshape(-1, 1)
            if i == 0:
                self.y_pred_all = y_pred
            else:
                self.y_pred_all = np.hstack((self.y_pred_all, y_pred))
            # 使用多数投票法作为最终预测结果
            y_pred_f = stats.mode(self.y_pred_all, axis=1).mode
            y_pred_f = y_pred_f.reshape(-1, 1)
            test_y = test_y.reshape(-1, 1)
            accuracy = np.sum((y_pred_f == test_y)) / test_y.size
            self.accuracy_all.append(accuracy)
        # 打印使用全部基学习器时的准确率
        print(f"T:{self.base_learner_times} random forest test accuracy:", self.accuracy_all[-1])
        # 返回所有模型的准确率列表,有需要时进行数据分析
        return self.accuracy_all


# 函数形式调用随机森林模型
def random_forest_function(train_x, train_y, test_x, test_y, base_learner_times=10):
    model = RandomForest(base_learner_times=base_learner_times)
    model.train(train_x, train_y)
    accuracy = model.predict(test_x, test_y)
    return accuracy
