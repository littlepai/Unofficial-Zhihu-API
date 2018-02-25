# coding: utf-8

import tensorflow as tf
import utils
from tensorflow.python.training import moving_averages


FLAGS = utils.FLAGS
num_classes = utils.num_classes


class LSTMOCR(object):
    def __init__(self, mode):
        self.mode = mode
        # 图像输入
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        
        # ctc_loss 需要的是稀疏矩阵
        self.labels = tf.sparse_placeholder(tf.int32)
        # 一维数组，大小[batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self._extra_train_ops = [] # 存储调整平滑均值，平滑方差的操作

    def build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        """
        构建模型，前两个卷积的卷积核size 分别是7，5 是很重要的，换成其他的效果会差很多
        """
        filters = [32, 64, 128, 128, FLAGS.max_stepsize]
        strides = [1, 2]

        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-0'):
                x = self._conv2d(self.inputs, 'cnn-0', 7, 1, filters[0], strides[0]) # 卷积
                x = self._batch_norm('bn0', x) # 批标准化
                x = self._leaky_relu(x, 0.01) # 非线性激活
                x = self._max_pool(x, 2, strides[0]) # 池化

            with tf.variable_scope('unit-1'):
                x = self._conv2d(x, 'cnn-1', 5, filters[0], filters[1], strides[0])
                x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])
            
            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2', 3, filters[1], filters[2], strides[0])
                x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-3'):
                x = self._conv2d(x, 'cnn-3', 3, filters[2], filters[3], strides[0])
                x = self._batch_norm('bn3', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

            with tf.variable_scope('unit-4'):
                x = self._conv2d(x, 'cnn-4', 3, filters[3], filters[4], strides[0])
                x = self._batch_norm('bn4', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, 2, strides[1])

        with tf.variable_scope('lstm'):
            shp = x.get_shape().as_list()
            x = tf.reshape(x, [-1, filters[4], shp[1]*shp[2]])
            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

            # 将rnn堆成2层深度
            stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

            # outputs是所有step的结果， state 是最后一个step的结果这里不需要
            outputs, _ = tf.nn.dynamic_rnn(stack, x, self.seq_len, dtype=tf.float32)

            # reshape 使其满足模型的step长度
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            W = tf.get_variable(name='W',
                                shape=[FLAGS.num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # reshape 使得最后一个维度是 num_classes
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)
        
        # ctc 损失函数，使用前后向算法和最大似然
        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.seq_len,
                                                                    merge_repeated=False) # 寻找最优路径
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1) # 解码
    
    # 卷积
    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b) #加上偏置，然后返回
    
    
    # 批标准化
    def _batch_norm(self, name, x):
        """批标准化."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]] #获取tensor的最后一个维度，后面的均值，方差都是这个维度
            
            # 标准化数据为均值为0方差为1之后，还有一个x=x*gamma+beta的调整
            # 这个会随着训练不断调整
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            
            # 训练的时候不断调整平滑均值，平滑方差
            # 预测的时候，回复权重使用的是训练过程中调整出来的平滑方差均值去做标准化
            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments') #获取批均值和方差，size[最后一个维度]
                
                # moving_mean, moving_variance 这两个name一定要让训练和预测的时候都相等，不然就没法恢复训练好的值了
                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                # mean的name一定要跟train的时候的一样 moving_mean
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                # variance的name一定要跟train的时候的一样 moving_variance
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                
                # 可视化
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            
            # 计算，标准化，最后一个值为误差，一般设置很小即可
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn
    
    # 变种Relu
    # Relu 简单而强大，方便求导
    # 非负区间的梯度为常数，一定程度上能够防止梯度消失问题
    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
