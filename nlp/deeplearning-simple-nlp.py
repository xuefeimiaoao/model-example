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

import numpy as np

class SentimentNetwork(object):
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.01):
        self.reviews = reviews
        self.labels = labels

    def pre_processing(self, reviews, labels):
        """
        统计reviews中出现的所有单词，并生成word2index
        :param reviews:
        :param labels:
        :return:
        """

        # 统计reviews中出现的所有单词
        review_vocab = set()
        for review in reviews.values:
            word = review[0].split(' ')
            review_vocab.update(word)

        self.review_vocab = list(review_vocab)

        # 统计labels中所有出现的label
        label_vocab = set()
        for label in labels.values:
            label_vocab.add(label[0])
        self.label_vocab = list(label_vocab)

        # 构建word2idx，给每个单词安排一个“门牌号”
        self.word2index = dict(zip(review_vocab, range(len(review_vocab))))

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
