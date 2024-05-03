import numpy as np


def fun1(data, i):
    TP = np.sum((load_data[:, 1] == 1.00) & (load_data[:, i] == 1.00))
    FN = np.sum((load_data[:, 1] == 1.00) & (load_data[:, i] == 0.00))
    FP = np.sum((load_data[:, 1] == 0.00) & (load_data[:, i] == 1.00))
    TN = np.sum((load_data[:, 1] == 0.00) & (load_data[:, i] == 0.00))
    return TP, FN, FP, TN


load_data = np.loadtxt('experiment_01_dataset_02.csv', delimiter=',')
n = load_data.shape[0]
TP1, FN1, FP1, TN1 = fun1(load_data, 2)
TP2, FN2, FP2, TN2 = fun1(load_data, 3)
TP3, FN3, FP3, TN3 = fun1(load_data, 4)
# 查准率
precision1 = TP1 / (TP1 + FP1)
precision2 = TP2 / (TP2 + FP2)
precision3 = TP3 / (TP3 + FP3)
# 查全率
recall1 = TP1 / (TP1 + FN1)
recall2 = TP2 / (TP2 + FN2)
recall3 = TP3 / (TP3 + FN3)
# F1
F1_1 = 2 * precision1 * recall1 / (precision1 + recall1)
F1_2 = 2 * precision2 * recall2 / (precision2 + recall2)
F1_3 = 2 * precision3 * recall3 / (precision3 + recall3)
# 输出
print('模型1:', ' TP:', TP1, ' FN:', FN1, ' FP:', FP1, ' TN:',
      TN1, ' 查准率:', precision1, ' 查全率:', recall1, ' F1:', F1_1)
print('模型2:', ' TP:', TP2, ' FN:', FN2, ' FP:', FP2, ' TN:',
      TN2, ' 查准率:', precision2, ' 查全率:', recall2, ' F1:', F1_2)
print('模型3:', ' TP:', TP3, ' FN:', FN3, ' FP:', FP3, ' TN:',
      TN3, ' 查准率:', precision3, ' 查全率:', recall3, ' F1:', F1_3)
