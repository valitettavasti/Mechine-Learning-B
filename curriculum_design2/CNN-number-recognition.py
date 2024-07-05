import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.nn import (
    Module, Sequential, Conv2d, ReLU, Linear, MaxPool2d, Dropout
)
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# 在有GPU的情况下使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# 模型结构
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Sequential(
            Conv2d(1, 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(stride=2, kernel_size=2)
        )

        self.dense = Sequential(
            Linear(14 * 14 * 128, 1024),
            ReLU(),
            Dropout(p=0.5),
            Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


model = Model().to(device)
cost = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters())  # Adam优化器的学习率默认为0.001
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)  # 采用不同的优化器比较结果
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.01)
# optimizer = torch.optim.Adamax(model.parameters(),lr=0.005)


number_epochs = 1

for epoch in range(number_epochs):
    running_loss = 0.0
    running_correct = 0
    all_predicted = []
    all_targets = []
    print('Epoch {}/{}'.format(epoch, number_epochs))
    print("-" * 10)
    # print('Epoch: {}/{}'.format(epoch, number_epochs))
    for data, target in data_loader_train:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        optimizer.zero_grad()
        loss = cost(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum((predicted == target).type(torch.FloatTensor))
    testing_correct = 0
    for data, target in data_loader_test:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        testing_correct += torch.sum((predicted == target).type(torch.FloatTensor))
        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    print("Loss: {:.4f}, Train Accuracy is {:.4f}%, Test Accuracy is {:.4f}%"
          .format(running_loss / len(data_train),
                  100 * running_correct / len(data_train),
                  100 * testing_correct / len(data_test)))
    conf_matrix = confusion_matrix(all_targets, all_predicted)
    # 计算每一个类别的查全率,查准率
    precision = precision_score(all_targets, all_predicted, average=None, zero_division=0)
    recall = recall_score(all_targets, all_predicted, average=None, zero_division=0)
    # 计算总体的查全率,查准率
    overall_precision = precision_score(all_targets, all_predicted, average='macro', zero_division=0)
    overall_recall = recall_score(all_targets, all_predicted, average='macro', zero_division=0)

    for i in range(10):
        print(f"类别{i} precision:{precision[i]:.4f} , recall:{recall[i]:.4f} , "
              f"F1:{2 * precision[i] * recall[i] / (precision[i] + recall[i]):.4f}")
    print(f"Overall Precision: {overall_precision:.4f} , Overall Recall: {overall_recall:.4f}"
          f"F1:{2 * overall_precision * overall_recall / (overall_precision + overall_recall):.4f}")
