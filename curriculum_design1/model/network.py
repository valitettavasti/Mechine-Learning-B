import numpy as np


class Network:
    """
    神经网络类,代表一个带有隐藏层的简单神经网络
    input_size:输入层大小
    hidden_size:隐藏层大小
    output_size:输出层大小
    theta,V,B,w:网络的权重和偏置
    alpha:网络训练的学习率，默认为0.1
    """

    def __init__(self, input_size, hidden_size, output_size, alpha=0.1) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.theta = np.random.randn(output_size, 1)
        self.V = np.random.randn(output_size, hidden_size)
        self.b = np.random.randn(hidden_size, 1)
        self.W = np.random.randn(hidden_size, input_size)
        self.alpha = alpha

    # sigmoid函数,使用静态函数
    @staticmethod
    def sigmoid(z_sigmoid):
        return 1 / (1 + np.exp(-z_sigmoid))

    def train(self, train_x, train_y):
        line = np.linspace(0, train_x.shape[0] - 1, train_x.shape[0])
        loss_array = []
        n = train_x.shape[0]
        train_y = train_y.astype(int)
        one_hot_h = np.zeros((n, self.output_size))
        one_hot_h[np.arange(n), train_y.reshape((n,))] = 1
        for j in range(100):  # epoch=100
            np.random.shuffle(line)  # 随机打乱索引数组
            for i in range(train_x.shape[0]):
                x = train_x[int(line[i]), :]
                x = x.reshape(-1, 1)
                h = one_hot_h[int(line[i]), :]
                h = h.reshape(-1, 1)
                z = np.dot(self.W, x) + self.b
                a = Network.sigmoid(z)
                t = np.dot(self.V, a) + self.theta
                y = Network.sigmoid(t)
                L = 1 / 2 * np.square(y - h)
                L_theta = (y - h) * y * (1 - y)
                L_V = np.dot(L_theta, a.T)
                L_b = np.dot(self.V.T, L_theta) * a * (1 - a)
                L_W = np.dot(L_b, x.T)
                self.theta = self.theta - self.alpha * L_theta
                self.V = self.V - self.alpha * L_V
                self.b = self.b - self.alpha * L_b
                self.W = self.W - self.alpha * L_W
            Loss = 0
            for i in range(train_x.shape[0]):
                # for i in range(10):
                x = train_x[i, :]
                x = x.reshape(-1, 1)
                h = one_hot_h[i, :]
                h = h.reshape(-1, 1)
                z = np.dot(self.W, x) + self.b
                a = Network.sigmoid(z)
                t = np.dot(self.V, a) + self.theta
                y = Network.sigmoid(t)
                L = 1 / 2 * np.square(y - h)
                loss = np.sum(L)
                Loss += loss
            loss_array.append(Loss)
        return loss_array

    # 网络的预测方法
    def predict(self, test_x, test_y):
        acc = 0
        predictions = []
        for i in range(test_x.shape[0]):
            x = test_x[i, :]
            x = x.reshape(-1, 1)
            h = test_y[i]
            z = np.dot(self.W, x) + self.b
            a = Network.sigmoid(z)
            t = np.dot(self.V, a) + self.theta
            y = Network.sigmoid(t)
            if np.argmax(y) == h:
                acc = acc + 1
            predictions.append(np.argmax(y))
        print('Network test accuracy', acc / test_x.shape[0])
        return predictions


def network_function(input_size, hidden_size, output_size, train_x, train_y, test_x, test_y,alpha=0.01):
    """
    神经网络的函数调用方法，采用于Network类中相同的代码,可以直接通过network_function调用
    """
    network = Network(input_size, hidden_size, output_size, alpha=alpha)
    loss = network.train(train_x, train_y)
    predict = network.predict(test_x, test_y)
    return loss, predict
