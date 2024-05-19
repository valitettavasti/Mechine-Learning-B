import pandas as pd
import numpy as np

# 读取数据
train_data = np.loadtxt('experiment_07_training_set.csv',
                        usecols=[1, 2, 3, 4],
                        delimiter=',',
                        skiprows=1)
train_tag = np.loadtxt('experiment_07_training_set.csv',
                       usecols=[-1],
                       delimiter=',',
                       skiprows=1,
                       dtype=str)
test_data = np.loadtxt('experiment_07_testing_set.csv',
                       usecols=[1, 2, 3, 4],
                       delimiter=',',
                       skiprows=1)
test_tag = np.loadtxt('experiment_07_testing_set.csv',
                      usecols=[-1],
                      delimiter=',',
                      skiprows=1,
                      dtype=str)

tag1_data = train_data[train_tag == "Iris-setosa", :]
tag2_data = train_data[train_tag == "Iris-versicolor", :]
tag3_data = train_data[train_tag == "Iris-virginica", :]
p_tag1 = tag1_data.shape[0] / train_data.shape[0]
p_tag2 = tag2_data.shape[0] / train_data.shape[0]
p_tag3 = tag3_data.shape[0] / train_data.shape[0]
print("P(Y=setosa:)", p_tag1)
print("P(Y=versicolor:)", p_tag2)
print("P(Y=virginica:)", p_tag3)
tag1_mean = np.mean(tag1_data, axis=0)
tag2_mean = np.mean(tag2_data, axis=0)
tag3_mean = np.mean(tag3_data, axis=0)
print("均值:")
print(tag1_mean)
print(tag2_mean)
print(tag3_mean)
tag1_std = np.std(tag1_data, axis=0)
tag2_std = np.std(tag2_data, axis=0)
tag3_std = np.std(tag3_data, axis=0)
print("标准差:")
print(tag1_std)
print(tag2_std)
print(tag3_std)


def calculate_pdf(test_data, tag_mean, tag_std, p_tag):
    test_tag = np.zeros([test_data.shape[0], 4])
    for i in range(4):
        test_tag[:, i] = 1 / (np.sqrt(2 * np.pi) * tag_std[i]) * np.exp(
            -(test_data[:, i] - tag_mean[i]) ** 2 / (2 * tag_std[i] ** 2))
    p_test_tag = np.ones(test_tag.shape[0])
    p_test_tag *= p_tag
    for i in range(4):
        p_test_tag *= test_tag[:, i]
    return p_test_tag


p_test_tag1 = calculate_pdf(test_data, tag1_mean, tag1_std, p_tag1)
p_test_tag2 = calculate_pdf(test_data, tag2_mean, tag2_std, p_tag2)
p_test_tag3 = calculate_pdf(test_data, tag3_mean, tag3_std, p_tag3)

p_test_tag1 = p_test_tag1.reshape(1, -1)
p_test_tag2 = p_test_tag2.reshape(1, -1)
p_test_tag3 = p_test_tag3.reshape(1, -1)

concatenated_array_col = np.hstack((p_test_tag1.T, p_test_tag2.T, p_test_tag3.T))
# print(concatenated_array_col)
index_max = np.argmax(concatenated_array_col, axis=1)
print(index_max)
test_tag = pd.Series(test_tag)
test_tag = pd.factorize(test_tag)[0]
print(test_tag)

acc = np.sum(index_max == test_tag)
print('Test Accuracy', acc / test_tag.shape[0])
