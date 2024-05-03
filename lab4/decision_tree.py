import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 读取数据
load_data_testing = np.loadtxt('experiment_04_testing_set.csv', delimiter=',')
load_data_training = np.loadtxt('experiment_04_training_set.csv', delimiter=',')

# 分割
X_train = load_data_training[:, :-1]
Y_train = load_data_training[:, -1]
X_test = load_data_testing[:, :-1]
Y_test = load_data_testing[:, -1]

# 训练模型
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
# tree = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=1)
# tree = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1)
# tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
tree.fit(X_train, Y_train)

# 画图
plot_tree(tree)
plt.show()

# 计算精度
predict = tree.predict(X_test)
accuracy = (np.sum((predict == Y_test).astype(int)) / np.size(Y_test, 0))
print(accuracy)
