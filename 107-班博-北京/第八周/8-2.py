import numpy as np
from tensorflow.keras.datasets import mnist


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # 初始化权重矩阵，W_i_h 和 W_h_o
        # 权重采用正态分布随机初始化，均值为0，标准差为节点传入链接数目的开方的倒数
        self.W_i_h = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.W_h_o = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数是sigmoid函数
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # 转换输入列表到二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 前向传播
        hidden_inputs = np.dot(self.W_i_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.W_h_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.W_h_o.T, output_errors)

        # 反向传播
        self.W_h_o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                       np.transpose(hidden_outputs))
        self.W_i_h += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        # 将输入列表转换为二维数组
        inputs = np.array(inputs_list, ndmin=2).T

        # 前向传播
        hidden_inputs = np.dot(self.W_i_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.W_h_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 神经网络参数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# 创建神经网络实例
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练神经网络
epochs = 5

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 将图像数据转换为神经网络的输入格式，并归一化
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

# 将标签转换为one-hot编码
train_labels = np.eye(output_nodes)[train_labels]

for e in range(epochs):
    for i in range(len(train_images)):
        n.train(train_images[i], train_labels[i])

# 测试神经网络
test_results = []
for img in test_images:
    outputs = n.query(img)
    label = np.argmax(outputs)
    test_results.append(label)

# 计算网络的性能
test_results = np.array(test_results)
correct_results = test_results == test_labels
accuracy = np.sum(correct_results) / test_labels.size
print(f'Accuracy: {accuracy}')
