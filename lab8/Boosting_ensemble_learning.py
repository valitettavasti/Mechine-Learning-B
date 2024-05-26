import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt

train_data = np.loadtxt('experiment_08_training_set.csv', delimiter=',')
test_data = np.loadtxt('experiment_08_testing_set.csv', delimiter=',')
train_x = train_data[:, 0:2]
train_y = train_data[:, 2]
test_x = test_data[:, 0:2]
test_y = test_data[:, 2]

w = np.full(train_data.shape[0], 1 / train_data.shape[0])

model_array = []
at_array = []
for i in range(1, 21):
    model = tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3)
    model.fit(train_x, train_y, sample_weight=w)
    predictions_train = model.predict(train_x)
    e_train = np.sum(w[predictions_train != train_y])
    at_train = 1 / 2 * np.log((1 - e_train) / e_train)
    w = w * np.exp(-train_y * at_train * predictions_train)
    w = w / np.sum(w)
    model_array.append(model)
    at_array.append(at_train)
predictions_array = np.zeros(100)
acc_array = []
for i in range(len(model_array)):
    predictions = model_array[i].predict(test_x)
    predictions_array = predictions_array + predictions * at_array[i]
    predictions_array_1 = np.sign(predictions_array)
    accuracy = np.sum((predictions_array_1 == test_y)) / test_y.size
    acc_array.append(accuracy)
    print(f"轮次{i + 1}:", accuracy)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.title("精度曲线图")
plt.plot(acc_array)
plt.show()
