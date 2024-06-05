import numpy as np
from sklearn import tree
from scipy import stats
from matplotlib import pyplot as plt

# 加载数据
training_data = np.loadtxt('experiment_09_training_set.csv', delimiter=',', skiprows=1)
testing_data = np.loadtxt('experiment_09_testing_set.csv', delimiter=',', skiprows=1)

# 数据分割及预处理
y_train = training_data[:, 0]
x_train = training_data[:, 1:] / 255.0
y_train = y_train.reshape(-1, 1)
y_test = testing_data[:, 0]
x_test = testing_data[:, 1:] / 255.0
y_test = y_test.reshape(-1, 1)

# 模型训练
model_all = []
for i in range(1, 21):
    choice_array = np.random.choice(x_train.shape[0], x_train.shape[0], replace=True)
    x_train_choice = x_train[choice_array, :]
    y_train_choice = y_train[choice_array]
    model = tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_features=50)
    model.fit(x_train_choice, y_train_choice)
    model_all.append(model)

# 计算精度
y_pred_all = []
accuracy_all = []
for i in range(len(model_all)):
    y_pred = model_all[i].predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    if i == 0:
        y_pred_all = y_pred
    else:
        y_pred_all = np.hstack((y_pred_all, y_pred))
    y_pred_f = stats.mode(y_pred_all, axis=1).mode
    y_pred_f = y_pred_f.reshape(-1, 1)
    accuracy = np.sum((y_pred_f == y_test)) / y_test.size
    accuracy_all.append(accuracy)
    print(f"T:{i+1} accuracy:", accuracy)

# 可视化
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.plot(range(1, 21), accuracy_all, marker='*', linewidth=0.8)
plt.xlabel("T")
plt.ylabel("精度")
plt.title("Bagging集成学习决策树数量T与精度曲线")
plt.show()
