import numpy as np

load_data = np.loadtxt('experiment_01_dataset_01.csv', delimiter=',')
x = load_data[:, 0]
y = load_data[:, 1]
prediction_1 = load_data[:, 2]
prediction_2 = load_data[:, 3]
prediction_3 = load_data[:, 4]
n = load_data.shape[0]
# 平均绝对误差
MAE1 = np.sum(np.abs(prediction_1 - y)) / n
MAE2 = np.sum(np.abs(prediction_2 - y)) / n
MAE3 = np.sum(np.abs(prediction_3 - y)) / n
print('MAE1:', MAE1, ' MAE2:', MAE2, ' MAE3', MAE3)
# 均方误差
MSE1 = np.sum(np.square(prediction_1 - y)) / n
MSE2 = np.sum(np.square(prediction_2 - y)) / n
MSE3 = np.sum(np.square(prediction_3 - y)) / n
print('MSE1:', MSE1, ' MSE2:', MSE2, ' MSE3', MSE3)
# 均方根误差
RMSE1 = np.sqrt(MSE1)
RMSE2 = np.sqrt(MSE2)
RMSE3 = np.sqrt(MSE3)
print('RMSE1:', RMSE1, ' RMSE2:', RMSE2, 'RMSE3:', RMSE3)
