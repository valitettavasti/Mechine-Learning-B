import numpy as np
import matplotlib.pyplot as plt

load_data_testing = np.loadtxt('experiment_02_testing_set.csv', delimiter=',')
load_data_training = np.loadtxt('experiment_02_training_set.csv', delimiter=',')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 求均值
x_mean = np.mean(load_data_training[:, 0])
# 对yi(xi-xbar)求和
sum_1 = np.sum(load_data_training[:, 1] * (load_data_training[:, 0] - x_mean))
# 对xi的平方求和
sum_2 = np.sum(np.square(load_data_training[:, 0]))
# 1/m的xi的和的平方
sum_3 = np.sum(load_data_training[:, 0]) ** 2 / np.size(load_data_training, 0)
# 斜率
k = sum_1 / (sum_2 - sum_3)
# 截距
b = np.sum(load_data_training[:, 1] - k * load_data_training[:, 0]) / np.size(load_data_training, 0)
print(k)
print(b)

plt.figure()
load_data_training = load_data_training[np.argsort(load_data_training[:, 0])]
load_data_testing = load_data_testing[np.argsort(load_data_testing[:, 0])]
plt.plot(load_data_training[:, 0], load_data_training[:, 1], label='training')
plt.plot(load_data_testing[:, 0], load_data_testing[:, 1], label='testing')
x = np.linspace(np.min(load_data_training[:, 0]), np.max(load_data_training[:, 0]), 50)
plt.plot(x, k * x + b, label='拟合曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

MSE = np.sum(np.square(load_data_testing[:, 1] - (load_data_testing[:, 0] * k + b))) / np.size(load_data_training,
                                                                                               0)
print('均方误差:', MSE)
