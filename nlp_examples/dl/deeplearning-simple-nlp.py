# Copyright (c) 2025 [xuefeimiaoao](https://github.com/xuefeimiaoao). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

import numpy as np
import nltk
import time
import pandas as pd

from nltk.corpus import stopwords
from pandas.core.interchange.dataframe_protocol import DataFrame


class SentimentNetwork(object):
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.01):
        np.random.seed(42)
        self.pre_processing(reviews, labels)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)


    def pre_processing(self, reviews: DataFrame, labels: DataFrame):
        """
        统计reviews中出现的所有单词，并生成word_index_dict
        :param reviews:
        :param labels:
        :return:
        """
        nltk.download('stopwords')

        # 统计reviews中出现的所有单词
        review_vocab = set()
        # 遍历reviews中的每一行: dataframe.values将df转换为numpy的二维数组，然后用for循环遍历每一行
        for review in reviews.values:
            # review[0]代表每一行的第一个元素，即review
            words = review[0].split(' ')
            for word in words:  # 逐个判断单词是否为停用词
                if word not in set(stopwords.words('english')):
                    review_vocab.add(word)

        self.review_vocab = list(review_vocab)

        # 统计labels中所有出现的label
        label_vocab = set()
        for label in labels.values:
            label_vocab.add(label[0])
        self.label_vocab = list(label_vocab)

        # 构建word2idx，给每个单词安排一个“门牌号”
        self.word_index_dict = dict(zip(review_vocab, range(len(review_vocab))))

    # todo 判断是否与我的NN相适合
    def kaiming_initialization(self, fan_in, fan_out=None):
        """
        Kaiming (He) initialization.

        Parameters:
        fan_in (int): Number of input units.
        fan_out (int, optional): Number of output units. If None, it is assumed that this is a fully connected layer.

        Returns:
        float: Standard deviation for weight initialization.
        """
        if fan_out is None:  # Fully connected layer
            scale = np.sqrt(2.0 / fan_in)
        else:  # Convolutional layer
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        return scale

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        初始化NN的参数
        :param input_nodes:
        :param hidden_nodes:
        :param output_nodes:
        :param learning_rate:
        :return:
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 如果权重初始化为0，那么神经网络一层的每个node都会相同
        # np.random.normal：这是 NumPy 库中的一个函数，用于生成服从正态分布的随机数。
        # 变量解释：
        # 0.0：这是正态分布的平均值（μ）。在这个例子中，我们生成均值为 0 的随机数，这是神经网络权重初始化时的一个常见选择，有助于保持输入的均值和方差在网络中传播时不会改变太多（这被称为“权重初始化”的一个原则）。
        # np.random.normal的第二个参数，正态分布的标准差。
        # (self.hidden_nodes, self.input_nodes)：这是一个元组，指定了输出数组的形状。在这个例子中，它表示我们想要生成一个形状为 (self.hidden_nodes, self.input_nodes)的数组，其中 self.input_nodes 是输入层节点的数量，self.hidden_nodes 是隐藏层节点的数量。
        self.weights_layer_1 = np.random.normal(0.0, self.kaiming_initialization(self.input_nodes, self.hidden_nodes), (self.hidden_nodes, self.input_nodes))
        self.weights_layer_2 = np.random.normal(0.0, self.kaiming_initialization(self.hidden_nodes, self.output_nodes), (self.output_nodes, self.hidden_nodes))

        # layer_0是词频向量
        self.layer_0 = np.zeros((self.input_nodes, 1))

    def update_input_layer(self, review):
        """
        更新输入层，todo 考虑大小写
        :param review:
        :return:
        """
        self.layer_0 *= 0
        for word in review.split(" "):
            if word in self.word_index_dict.keys():
                # layer_0是词频向量
                self.layer_0[self.word_index_dict[word]][0] += 1

    def relu(self, x):
        """
        ReLU activation function
        :param x:
        :return:
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of ReLU function
        :param x:
        :return:
        """
        return 1 * (x > 0)

    def sigmoid(self, x):
        """
        Sigmoid function
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        """
        Derivative of sigmoid function
        :param output: 即y hat。 即y hat = 1 / (1 + np.exp(-x))
        :return:
        """
        return output * (1 - output)

    def get_target_for_label(self, label):
        return label

    def train(self, training_reviews: DataFrame, training_label: DataFrame):
        assert (len(training_reviews) == len(training_label))
        correct_so_far = 0
        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews.iloc[i, 0]
            label = training_label.iloc[i, 0]

            self.update_input_layer(review)

            ###
            # Forward propagation:
            # Input a^[l-1]
            # Output a^[l], cache z^[l]
            ###

            # np.dot(a, b)：计算两个矩阵的乘积，即a和b的对应元素相乘，然后求和。
            # 变量解释：
            # self.layer_0：这通常代表神经网络的输入层或前一层的输出。它是一个二维数组（或矩阵），其中包含了当前层需要处理的输入数据。在这个上下文中，self.layer_0 的形状可能是 (n_samples, n_features)，其中 n_samples 是样本数量，n_features 是特征数量。
            # self.weights_layer_1：这是连接输入层（或前一层）到当前层的权重矩阵。它的形状通常是 (n_features, n_hidden_units)，其中 n_hidden_units 是当前层（隐藏层）的神经元数量。
            # layer_1_i：这代表当前层（第一层隐藏层）的输入。计算完成后，它将包含加权后的输入数据，准备传递给激活函数（如果有的话）。
            layer_1_i = np.dot(self.weights_layer_1, self.layer_0)
            layer_1_o = self.relu(layer_1_i)

            layer_2_i = np.dot(self.weights_layer_2, layer_1_o)
            layer_2_o = self.sigmoid(layer_2_i)

            ##
            # Backward propagation:
            # Input da^[l]
            # Output da^[l-1], dw^[l], db^[l]
            ###

            # da^[2] = (1-y)/(1-a) - y/a                                          a^[l] (n^[l], 1)
            # dz^[2] = da^[2] * sigmoid_output_2_derivative(a^[2])                z^[l] (n^[l], 1)
            # dw^[2] = dz^[2] * a^[1].T. todo ???需要转秩 难道吴恩达课程上错了？       w^[l] (n^[l], n^[l-1])
            # db^[2] = dz^[2]                                                     b^[l] (n^[l], 1)
            # da^[1] = w^[2].T * dz^[2]

            da_2 = (1 - self.get_target_for_label(label)) / (1 - layer_2_o) - self.get_target_for_label(label) / layer_2_o
            dz_2 = da_2 * self.sigmoid_output_2_derivative(layer_2_o) # todo why？逐元素相乘
            print(f'da_2 shape: {da_2.shape}')
            print(f'dz_2 shape: {dz_2.shape}')
            print(f'layer_1_o shape: {layer_1_o.shape}')
            dw_2 = np.dot(dz_2, layer_1_o.T)
            db_2 = dz_2
            da_1 = np.dot(self.weights_layer_2.T, dz_2)

            print(f'dw_2 shape: {dw_2.shape}')
            print(f'db_2 shape: {db_2.shape}')
            print(f'da_1 shape: {da_1.shape}')
            print(f'weights_layer_2 shape: {self.weights_layer_2.shape}')


            self.weights_layer_2 -= self.learning_rate * dw_2

            # da^[1] = w^[2].T * dz^[2]
            # dz^[1] = da^[1] * relu_derivative(a^[1])
            # dw^[1] = dz^[1] * a^[0].T
            # db^[1] = dz^[1]

            dz_1 = da_1 * self.relu_derivative(layer_1_o)
            dw_1 = np.dot(dz_1, self.layer_0.T)
            db_1 = dz_1
            self.weights_layer_1 -= self.learning_rate * dw_1


            # 判断预测结果是否正确
            if layer_2_o >= 0.5 and label == 1:
                correct_so_far += 1
            elif layer_2_o < 0.5 and label == 0:
                correct_so_far += 1

            elapsed = float(time.time() - start)
            review_per_sec = i / elapsed if elapsed > 0 else 0

            print("Progress: {0:.4}%".format((i + 1) / len(training_reviews) * 100))
            print("Speed(reviews/sec): {0:.5}".format(review_per_sec))
            print("Correct: {0}/{1}".format(correct_so_far, i+1))
            print("Accuracy: {0:.4}%".format(correct_so_far / float(i+1) * 100))

            if i % 2500 == 0:
                print("")

    def test(self, test_reviews, test_labels):
        assert (len(test_reviews) == len(test_labels))

        correct = 0

        start = time.time()

        for i in range(len(test_reviews)):
            review = test_reviews.iloc[i, 0]
            label = test_labels.iloc[i, 0]

            pred = self.run(review)
            if pred == label:
                correct += 1

            elapsed = float(time.time() - start)
            review_per_sec = i / elapsed if elapsed > 0 else 0

            print("\rProgress: {0:.4}%".format((i + 1) / len(test_reviews) * 100))


        print("")
        print("Test complete")
        print("Accuracy: {0}%".format(correct / float(len(test_reviews)) * 100))

    def run(self, review):
        self.update_input_layer(review)

        layer_1_i = np.dot(self.weights_layer_1, self.layer_0)
        layer_1_o = self.relu(layer_1_i)

        layer_2_i = np.dot(self.weights_layer_2, layer_1_o)
        layer_2_o = self.sigmoid(layer_2_i)

        if layer_2_o >= 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    splits = {'train': 'train.parquet', 'validation': 'validation.parquet', 'test': 'test.parquet'}
    df = pd.read_parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/" + splits["train"])
    reviews = df[['text']]
    labels = df[['label']]
    # mlp = SentimentNetwork(reviews, labels, learning_rate=0.1)
    # mlp.train(reviews[:-1000], labels[:-1000])

    mlp = SentimentNetwork(reviews, labels, learning_rate=0.01)
    mlp.train(reviews, labels)

    df_test = pd.read_parquet("hf://datasets/cornell-movie-review-data/rotten_tomatoes/" + splits["test"])
    test_reviews = df_test[['text']]
    test_labels = df_test[['label']]
    mlp.test(test_reviews, test_labels)




