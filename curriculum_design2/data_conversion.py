import numpy as np


def torch_data_to_sklearn(data_loader):
    images = []
    labels = []
    for data, target in data_loader:
        images.append(data.view(data.size(0), -1).numpy())
        labels.append(target.numpy())
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels).ravel()  # 将标签展平为一维数组
    return images, labels
