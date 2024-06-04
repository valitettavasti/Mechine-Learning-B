import numpy as np
import random
from matplotlib import pyplot as plt

data = np.loadtxt('experiment_10_training_set.csv', delimiter=',')

loss = []
for m in range(10):
    center_index = random.sample(range(data.shape[0]), m + 1)
    center = data[center_index, 0:2]
    tag = np.zeros([data.shape[0], 1])
    data_tag = np.hstack((data, tag))
    for k in range(100):
        for i in range(data_tag.shape[0]):
            min_distance = np.inf
            min_tag = None
            for j in range(center.shape[0]):
                distance = np.square(data[i][0] - center[j][0]) + np.square(data[i][1] - center[j][1])
                if distance < min_distance:
                    min_distance = distance
                    min_tag = j
            data_tag[i][2] = min_tag
        for j in range(center.shape[0]):
            tag_array = data_tag[np.where(data_tag[:, -1] == j)[0]]
            center[j] = np.mean(tag_array[:, 0:2], axis=0)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'teal', 'darkorange', 'purple', 'black']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    for j in range(center.shape[0]):
        tag_array = data_tag[np.where(data_tag[:, -1] == j)[0]]
        plt.scatter(tag_array[:, 0], tag_array[:, 1], c=colors[j], alpha=0.4)
        plt.title(f"类别数:k={m + 1}")
    plt.show()
    loss_array = np.zeros([data_tag.shape[0], 1])
    center_indices = data_tag[:, 2].astype(int)
    distances_squared = ((data_tag[:, :2] - center[center_indices, :]) ** 2).sum(axis=1)
    loss_number = np.sum(distances_squared)
    loss = np.append(loss, loss_number)
print(loss)
plt.plot(range(1, 11), loss, marker='o')
plt.xlabel("k")
plt.ylabel("loss")
plt.title("loss随k值增加的变化曲线图")
plt.show()
