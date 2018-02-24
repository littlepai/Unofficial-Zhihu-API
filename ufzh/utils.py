# coding: utf-8

import os
import numpy as np
import tensorflow as tf
#import cv2
from PIL import Image

# +-* + () + 10 digit + blank + space
num_classes = 38#3 + 2 + 10 + 1 + 1

maxPrintLen = 100

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 150, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('max_stepsize', 64, 'max stepsize in lstm, as well as '
                                                'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 500, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 1000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')


FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

import string
charset = string.digits + string.ascii_lowercase#'0123456789+-*()'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


class DataIterator:
    def __init__(self, data_dir):
        self.image = [] # 所有的图片都加载进内存待等待提取，加载进内存当然是为了速度了
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                # im = np.array(Image.open(image_name)).astype(np.float32)/255.
                im = np.array(Image.open(image_name).convert("L")).astype(np.float32)/255.
                # im = np.array(Image.open(image_name).convert("L").point(lambda x: 0 if x < 150 else 1)).astype(np.float32)
                # im = cv2.imread(image_name, 0).astype(np.float32)/255.
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split(os.sep)[-1].split('_')[1].split('.')[0] # code 是验证码
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)] # code转成[1,2,3,4] 字码列表
                self.labels.append(code)
    
    # 使size方法变成属性，调用的时候self.size即可，不用调用self.size() #这里体现不出@property的优点
    @property
    def size(self):
        return len(self.labels)
    
    # 给定index, 抽取labels
    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    # 给定index, 得到一个批次的训练数据
    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 分片的序列长度，因为验证码图片序列长度都是一样的，不像句子有长有短
            # 所以这里的长度都是一样的
            lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch) # 转成稀疏矩阵

        return batch_inputs, batch_seq_len, batch_labels

# 对比解码得到的label和真实label，计算正确率
def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """
    根据[[1,2,3,4],[5,2,6,5],...] 这种形式，生成稀疏矩阵
    稀疏矩阵由三个元素的tuple组成，即(indices, values, shape)
    indices和values的都是一个列表，列表元素刚好一一对应，
    一个代表坐标位置，一个代表这个位置的值，其中indices是一个
    [(0,1),(0,2),...(10,3),(10,4),...]这样的形式的列表，指示了
    对应的values的值在密集矩阵的坐标，values 是[1,2,3,...,100,...]
    这样的形式，最后一个shape描述密集矩阵的shape
    
    示例：
        indices = [(0,1),(0,2),(0,3),(1,1),(1,3),(2,2)]
        values = [1,2,3,4,5,6]
        shape = [4,3]
        
        则对应的密集矩阵就是
            0 1 2 3
            0 4 0 5
            0 0 6 0
    
    参数:
        sequences: 一个列表，列表里面是每个验证码的码字列表
    
    返回:
        (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
