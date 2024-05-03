import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read data from files
train_data = pd.read_csv('experiment_05_training_set.csv')
test_data = pd.read_csv('experiment_05_testing_set.csv')
train_data = np.array(train_data)
test_data = np.array(test_data)
train_x = train_data[:, 1:] / 255
train_y = train_data[:, 0]
test_x = test_data[:, 1:] / 255
test_y = test_data[:, 0]
# one-hot encoding
n = train_x.shape[0]
one_hot_h = np.zeros((n, 10))
one_hot_h[np.arange(n), train_y.reshape((n,))] = 1
# initialization
theta = np.random.randn(10, 1)
V = np.random.randn(10, 12)
b = np.random.randn(12, 1)
W = np.random.randn(12, 784)

alpha = 0.01


# sigmoid函数
def sigmoid(z_sigmoid):
    return 1 / (1 + np.exp(-z_sigmoid))


line = np.linspace(0, 29999, 30000)  # 创造索引数组
loss_array = []
for j in range(100):
    np.random.shuffle(line)   #随机打乱索引数组
    for i in range(train_x.shape[0]):
        x = train_x[int(line[i]), :]
        x = x.reshape(-1, 1)
        h = one_hot_h[int(line[i]), :]
        h = h.reshape(-1, 1)
        z = np.dot(W, x) + b
        a = sigmoid(z)
        t = np.dot(V, a) + theta
        y = sigmoid(t)
        L = 1 / 2 * np.square(y - h)
        L_theta = (y - h) * y * (1 - y)
        L_V = np.dot(L_theta, a.T)
        L_b = np.dot(V.T, L_theta) * a * (1 - a)
        L_W = np.dot(L_b, x.T)
        theta = theta - alpha * L_theta
        V = V - alpha * L_V
        b = b - alpha * L_b
        W = W - alpha * L_W
    Loss = 0
    for i in range(train_x.shape[0]):
        # for i in range(10):
        x = train_x[i, :]
        x = x.reshape(-1, 1)
        h = one_hot_h[i, :]
        h = h.reshape(-1, 1)
        z = np.dot(W, x) + b
        a = sigmoid(z)
        t = np.dot(V, a) + theta
        y = sigmoid(t)
        L = 1 / 2 * np.square(y - h)
        loss = np.sum(L)
        Loss += loss
    loss_array.append(Loss)
plt.plot(loss_array)
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.title("损失迭代曲线(学习率为0.01)")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.show()

acc = 0

for i in range(test_data.shape[0]):
    x = test_x[i, :]
    x = x.reshape(-1, 1)
    h = test_y[i]
    z = np.dot(W, x) + b
    a = sigmoid(z)
    t = np.dot(V, a) + theta
    y = sigmoid(t)
    if np.argmax(y) == h:
        acc = acc + 1
print('Test Accuracy', acc / test_data.shape[0])
