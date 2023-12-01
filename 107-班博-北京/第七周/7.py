import numpy as np
from sklearn import preprocessing

# 示例数据
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 最大-最小归一化
min_max_scaler = preprocessing.MinMaxScaler()
min_max_normalized_data = min_max_scaler.fit_transform(data)
print("Min-Max normalized data:\n", min_max_normalized_data)

# Z-score标准化
standardized_data = preprocessing.scale(data)
print("Z-score standardized data:\n", standardized_data)

# 均值归一化
mean_normalized_data = preprocessing.normalize(data, norm='l2')
print("Mean normalized data:\n", mean_normalized_data)

# Sigmoid归一化
sigmoid_normalized_data = 1 / (1 + np.exp(-data))
print("Sigmoid normalized data:\n", sigmoid_normalized_data)
