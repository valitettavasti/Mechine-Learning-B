from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from data_conversion import torch_data_to_sklearn

# 下载数据
data_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
)
data_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
)

# 将数据分批次并随机打乱
data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
data_loader_test = DataLoader(data_test, batch_size=64, shuffle=True)

x_train, y_train = torch_data_to_sklearn(data_loader_train)
x_test, y_test = torch_data_to_sklearn(data_loader_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)

y_test = y_test.reshape(-1, 1)
prediction = prediction.reshape(-1, 1)
print(np.sum(y_test == prediction)/len(y_test))
conf_matrix = confusion_matrix(y_test, prediction)
overall_precision = precision_score(y_test, prediction, average='macro', zero_division=0)
overall_recall = recall_score(y_test, prediction, average='macro', zero_division=0)
print(f"Overall Precision: {overall_precision:.4f} , Overall Recall: {overall_recall:.4f}"
      f"F1:{2 * overall_precision * overall_recall / (overall_precision + overall_recall):.4f}")
f1 = f1_score(y_test, prediction, average='macro', zero_division=0)
print(f"F1 Score: {f1:.4f}")