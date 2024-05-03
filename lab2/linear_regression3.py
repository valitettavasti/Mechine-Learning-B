import numpy as np
import matplotlib.pyplot as plt

load_data_testing = np.loadtxt('experiment_02_testing_set.csv', delimiter=',')
load_data_training = np.loadtxt('experiment_02_training_set.csv', delimiter=',')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

X = np.ones((np.size(load_data_training, 0), 3))
X[:, 0] = load_data_training[:, 0]
X[:, 1] = np.square(load_data_training[:, 0])
w = np.linalg.inv(X.T @ X) @ X.T @ load_data_training[:, 1]

print(w)

plt.figure()
load_data_training = load_data_training[np.argsort(load_data_training[:, 0])]
load_data_testing = load_data_testing[np.argsort(load_data_testing[:, 0])]
plt.plot(load_data_training[:, 0], load_data_training[:, 1], label='training')
plt.plot(load_data_testing[:, 0], load_data_testing[:, 1], label='testing')
x = np.linspace(np.min(load_data_training[:, 0]), np.max(load_data_training[:, 0]), 50)
plt.plot(x, w[0] * x + w[1] * x ** 2 + w[2], label='拟合曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

MSE = (np.sum(np.square(
    load_data_testing[:, 1] - (load_data_testing[:, 0] * w[0] + load_data_testing[:, 0] ** 2 * w[1] + w[2]))) /
       np.size(load_data_training, 0))
print('均方误差:', MSE)
