import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import DataBatch as DataBatch
import re

def viterbi(nodes):
    """
    维特比译码：除了第一层以外，每一层有4个节点。
    计算当前层（第一层不需要计算）四个节点的最短路径：
       对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
       上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
       paths 采用字典的形式保存（路径：路径长度）。
       一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
    """
    zy = {
        'BE': 0.828739514282,
        'BM': 0.171260485718,
        'EB': 0.592369623847,
        'ES': 0.407630376153,
        'ME': 0.504871829789,
        'MM': 0.495128170211,
        'SB': 0.623266273388,
        'SS': 0.376733726612
    }
    paths = {'B': nodes[0]['B'], 'S': nodes[0]['S']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[
                        path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]  # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class BiLSTMWordSeparator(object):
    def __init__(self, save_path='./bi-lstm/bi-lstm.ckpt'):

        self.input_size = self.embedding_size = 100  # 每一个字向量的维数
        self.word_size = 3000  # 字数
        self.hidden_size = 128
        self.class_size = 5  # 每个字有4种标签, 加上0一共5种类别
        self.timestep_size = 32  # 每句话的长度
        self.layer_num = 2  # lstm层数
        self.max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

        self.save_path = save_path

        self.word_2_index_dict = dict()  # 每个汉字和one-hot对照表
        self.index_2_word_dict = dict()
        self.sign = {'B': 1, 'M': 2, 'E': 3, 'S': 4}
        #  导入字典
        with open('character_one_hot.txt') as f:
            for line in f:
                l = line.split()
                self.index_2_word_dict[int(l[1])] = l[0]
                self.word_2_index_dict[l[0]] = int(l[1])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.Graph()

        self.x_inputs, self.y_inputs, self.accuracy, self.cost, self.train_op, self.predict = self.create_graph()

        self.sess = sess = tf.Session(config=config, graph=self.graph)

        with self.graph.as_default():
            sess.run(tf.global_variables_initializer())
            #  导入字向量
            # print('importing character embedding')
            self.saver = saver = tf.train.Saver()
            saver.restore(sess, self.save_path)
            # embedding_saver = tf.train.Saver({'embeddings': self.graph.get_tensor_by_name('embeddings:0')})
            # embedding_saver.restore(sess, './character_embedding/embeddings')
            # print("%d %s " % (10, self.index_2_word_dict[10]))
            # print(str(sess.run(tf.nn.embedding_lookup(self.graph.get_tensor_by_name('embeddings:0'), 10))))

    def trunc_list(self, l):
        if len(l) >= self.timestep_size:
            return l[0: self.timestep_size]
        else:
            return np.append(l, np.zeros((self.timestep_size - len(l))))

    def read_data(self, data_path):
        #  导入训练数据
        raw_data = [[], []]
        with open(data_path) as f:
            for line in f:
                if line == '\n':
                    continue
                assert (len(line) - 1) % 3 == 0  # 去掉换行符
                line_data = [[], []]
                for i in range(len(line) // 3):
                    if line[i * 3] in self.word_2_index_dict:
                        word_index = self.word_2_index_dict[line[i * 3]]
                    else:
                        word_index = 0
                    line_data[0].append(word_index)
                    line_data[1].append(self.sign[line[i * 3 + 2]])
                line_data[0] = self.trunc_list(line_data[0])  # 把长度限制在32
                line_data[1] = self.trunc_list(line_data[1])
                raw_data[0].append(line_data[0])
                raw_data[1].append(line_data[1])
        return raw_data

    #  LSTM层
    def cell(self, keep_prob):
        cell = rnn.BasicLSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    def create_graph(self):
        with self.graph.as_default():
            lr = tf.placeholder(tf.float32, [], name='learning_rate')
            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            embeddings = tf.Variable(np.zeros([self.word_size, self.embedding_size]), dtype=tf.float32,
                                     name='embeddings', trainable=False)

            x_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='x_input')
            y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')
            inputs = tf.nn.embedding_lookup(embeddings, x_inputs)

            #  多层LSTM
            cell_fw = rnn.MultiRNNCell([self.cell(keep_prob) for _ in range(self.layer_num)])
            cell_bw = rnn.MultiRNNCell([self.cell(keep_prob) for _ in range(self.layer_num)])
            #  初始状态
            initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            #  将输入分解成合适的格式
            # inputs = tf.unstack(inputs, num=None, axis=1)
            # inputs = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2]])
            inputs = tf.transpose(inputs, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, 100])
            inputs = tf.split(inputs, 32)
            #  计算
            outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                         initial_state_fw=initial_state_fw,
                                                         initial_state_bw=initial_state_bw,
                                                         dtype=tf.float32)
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size * 2])
            #  再接一层传统的全连接层
            softmax_w = weight_variable([self.hidden_size * 2, self.class_size])
            softmax_b = bias_variable([self.class_size])
            y_pred = tf.matmul(output, softmax_w) + softmax_b

            correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
        return x_inputs, y_inputs, accuracy, cost, train_op, y_pred

    def test_epoch(self, data):
        """Testing or valid."""
        _batch_size = 128
        fetches = [self.accuracy, self.cost]
        batch_num = data.size // _batch_size
        _costs = 0.0
        _accs = 0.0
        for i in range(batch_num):
            X_batch, y_batch = data.next_batch(_batch_size)
            feed_dict = {self.x_inputs: X_batch, self.y_inputs: y_batch,
                         self.graph.get_tensor_by_name('learning_rate:0'): 1e-4,
                         self.graph.get_tensor_by_name('keep_prob:0'): 1.0,
                         self.graph.get_tensor_by_name('batch_size:0'): _batch_size}
            _acc, _cost = self.sess.run(fetches, feed_dict)
            _accs += _acc
            _costs += _cost
        mean_acc = _accs / batch_num
        mean_cost = _costs / batch_num
        return mean_acc, mean_cost

    def train(self, train_data_path='training_data.txt', target_acc=0.94):
        raw_data = self.read_data(train_data_path)
        x_train, x_test, y_train, y_test = train_test_split(raw_data[0], raw_data[1], test_size=0.2, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        train_data = DataBatch.DataBatch(x_train, y_train, True)
        test_data = DataBatch.DataBatch(x_test, y_test)
        valid_data = DataBatch.DataBatch(x_valid, y_valid)

        tr_batch_size = 128
        lr_decay = 0.85
        max_max_epoch = 6  # 最多重复训练6次
        tr_batch_num = train_data.size // tr_batch_size

        with self.sess.as_default():
            finish = False
            _lr = 1e-2
            for epoch in range(max_max_epoch):
                _lr = _lr * (lr_decay ** epoch)
                print("epoch: %d, learning rate:%g " % (epoch + 1, _lr))
                start_time = time.time()
                _costs = 0.0
                _accs = 0.0
                show_accs = 0.0
                show_costs = 0.0
                for batch in range(tr_batch_num):
                    fetches = [self.accuracy, self.cost, self.train_op]
                    x_batch, y_batch = train_data.next_batch(tr_batch_size)
                    feed_dict = {self.x_inputs: x_batch, self.y_inputs: y_batch,
                                 self.graph.get_tensor_by_name('learning_rate:0'): _lr,
                                 self.graph.get_tensor_by_name('keep_prob:0'): 0.5,
                                 self.graph.get_tensor_by_name('batch_size:0'): tr_batch_size}
                    _acc, _cost, _ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
                    _accs += _acc
                    _costs += _cost
                    show_accs += _acc
                    show_costs += _cost
                    if (batch + 1) % 5 == 0:
                        valid_acc, valid_cost = self.test_epoch(valid_data)
                        print('training %d/%d acc=%g, cost=%g, valid acc=%g, cost=%g' %
                              (batch + 1, tr_batch_num, show_accs / 5, show_costs / 5, valid_acc, valid_cost))
                        if valid_acc >= target_acc:
                            finish = True
                            break
                        show_accs = 0
                        show_costs = 0
                mean_acc = _accs / tr_batch_num
                mean_cost = _costs / tr_batch_num

                print('training %d, acc=%g, cost=%g ' % (train_data.size, mean_acc, mean_cost))
                print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' %
                      (train_data.size, mean_acc, mean_cost, time.time() - start_time))
                if finish:
                    break
            save_path = self.saver.save(self.sess, self.save_path)
            print('model saved at ', save_path)
            test_acc, test_cost = self.test_epoch(test_data)
            print('Test result: test %d, acc=%g, cost=%g' % (test_data.size, test_acc, test_cost))

    def separate_line(self, sentence):
        sequence = np.asarray([self.trunc_list([self.word_2_index_dict.get(i, 0) for i in sentence])])
        feed_dict = {self.x_inputs: sequence,
                     self.graph.get_tensor_by_name('keep_prob:0'): 1,
                     self.graph.get_tensor_by_name('batch_size:0'): 1}
        result = self.sess.run(self.predict, feed_dict=feed_dict)[0: len(sentence)]
        nodes = [dict(zip(['B', 'M', 'E', 'S'], each[1:])) for each in result]
        tags = viterbi(nodes)
        words = []
        for i in range(len(sentence)):
            if tags[i] in ['S', 'B']:
                words.append(sentence[i])
            else:
                words[-1] += sentence[i]
        return '/'.join(words)

    def separate(self, sentence):
        pattern = re.compile(r'([A-za-z0-9]+|[,.\';!，。？！、”“‘’\s])')
        sp = pattern.split(sentence)
        result = []
        for e in sp:
            if len(e) == 0:
                continue
            if pattern.fullmatch(e) == None:
                result.append(self.separate_line(e))
            else:
                result.append(e)
        return '/'.join(result)



if __name__ == '__main__':
    separator = BiLSTMWordSeparator()
    # separator.train()
    sentence = u'其实之前就已经用过 LSTM 了，是在深度学习框架 keras 上直接用的，但是到现在对LSTM详细的网络结构还是不了解，心里牵挂着难受呀！今天看了 tensorflow 文档上面推荐的这篇博文，看完这后，焕然大悟，对 LSTM 的结构理解基本上没有太大问题。此博文写得真真真好！！！为了帮助大家理解，也是怕日后自己对这些有遗忘的话可以迅速回想起来，所以打算对原文写个翻译。首先声明，由于本人水平有限，如有321翻译不好或理解有误的多多指出！此外，本译文也不是和原文一字一句对应的，为了方便理解可能会做一些调整和修改。）123'
    print(separator.separate(sentence))
