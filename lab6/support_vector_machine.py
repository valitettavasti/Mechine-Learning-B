import numpy as np
from scipy.optimize import LinearConstraint, minimize
import matplotlib.pyplot as plt

training_data = np.loadtxt('experiment_06_training_set.csv', delimiter=',')
testing_data = np.loadtxt('experiment_06_testing_set.csv', delimiter=',')
train_point = training_data[:, 0:2]
train_tag = training_data[:, 2].reshape(-1, 1)

# wb = np.zeros((3,))
wb = np.array([np.random.rand() for _ in range(3)])
y_x = train_point * train_tag
a = np.hstack((y_x, train_tag))


def objective(w):
    return 0.5 * (w[0] ** 2 + w[1] ** 2)


iteration = 1
loss_iteration = []
loss_iteration.append(objective(wb))


def print_loss(intermediate_result):
    global iteration
    global loss_iteration
    parameter = intermediate_result  # .x
    loss_iteration.append(objective(parameter))
    print('iteration', iteration, 'loss', objective(parameter))
    iteration += 1


lc = LinearConstraint(a, lb=1, ub=np.inf)
aa = minimize(objective, wb, constraints=lc, callback=print_loss)
print(aa)
print(aa.x)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.plot(loss_iteration)
plt.title("损失函数迭代曲线")
plt.show()

indices_positive = np.where(train_tag == 1)[0]
indices_negative = np.where(train_tag == -1)[0]
positive_points = train_point[indices_positive]
negative_points = train_point[indices_negative]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', label='正例', alpha=0.6)
plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', label='负例', alpha=0.6)
plt.title("训练集分类超平面图")
x = np.linspace(-1, 5, 400)
y = - aa.x[0] / aa.x[1] * x - aa.x[2] / aa.x[1]
plt.plot(x, y, lw=2.0)
plt.legend()
plt.show()

testing_point = testing_data[:, 0:2]
testing_tag = testing_data[:, 2]
predict_tag = aa.x[0] * testing_point[:, 0] + aa.x[1] * testing_point[:, 1] + aa.x[2]
predict_tag = np.sign(predict_tag)
predict_tag = np.where(predict_tag == 0, 1, predict_tag)
correct_predictions = np.sum(testing_tag == predict_tag)
print('Test Accuracy', correct_predictions / predict_tag.shape[0])

indices_positive = np.where(testing_tag == 1)[0]
indices_negative = np.where(testing_tag == -1)[0]
positive_points = testing_point[indices_positive]
negative_points = testing_point[indices_negative]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', label='正例', alpha=0.6)
plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', label='负例', alpha=0.6)
plt.title("测试集分类超平面图")
x = np.linspace(-1, 5, 400)
y = - aa.x[0] / aa.x[1] * x - aa.x[2] / aa.x[1]
plt.plot(x, y, lw=2.0)
plt.legend()
plt.show()
