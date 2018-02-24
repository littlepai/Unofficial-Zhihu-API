# coding: utf-8

import datetime
import logging
import os
import math
import time
from PIL import Image
import numpy as np
import tensorflow as tf

from ufzh import orcmodel
from ufzh import utils
import helper

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    #加载模型类
    model = orcmodel.LSTMOCR(mode)
    #创建模型，这一步，运算图和训练操作等都已经具备
    model.build_graph()

    print('loading train data, please wait---------------------')
    train_feeder = utils.DataIterator(data_dir=train_dir) # 准备训练数据
    print('get image: ', train_feeder.size)

    print('loading validation data, please wait---------------------')
    val_feeder = utils.DataIterator(data_dir=val_dir) # 准备验证数据
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  # 训练样本个数
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # 一轮有多少批次 example: 100000/100

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # 一轮有多少批次 example: 10000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.device('/cpu:0'):
        # ConfigProto 用于配置Session
        # allow_soft_placement=True 意思是如果你指定的设备不存在，允许TF自动分配设备
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            # max_to_keep 用于指定保存最近的N个checkpoint
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) # summary 可视化
            # 根据配置是否恢复权重
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # 恢复的时候global_step也会被恢复
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                shuffle_idx = np.random.permutation(num_train_samples) # 乱序训练样本的index，达到SGD
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()

                # 开始一轮的N批训练
                for cur_batch in range(num_batches_per_epoch):
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    
                    # 生成训练批次的index
                    indexs = [shuffle_idx[i % num_train_samples] for i in
                              range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                    batch_inputs, batch_seq_len, batch_labels = \
                        train_feeder.input_index_generate_batch(indexs)
                    
                    # 填充placeholder
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels,
                            model.seq_len: batch_seq_len}

                    # 开始run，记录可视化数据，计算成本值，获取global_step，并训练
                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)
                    
                    # batch_cost是一个均值，这里计算一个batch的cost
                    train_cost += batch_cost * FLAGS.batch_size

                    train_writer.add_summary(summary_str, step) # run merge_all的到一个summery信息，然后写入，用于可视化

                    # 保存checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)

                    # 验证
                    if step % FLAGS.validation_steps == 0:
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        for j in range(num_batches_per_epoch_val):
                            # 按SGD验证一个最小批
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                          range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                            val_inputs, val_seq_len, val_labels = \
                                val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels,
                                        model.seq_len: val_seq_len}

                            # 解码，并获得当前学习率
                            dense_decoded, lr = sess.run([model.dense_decoded, model.lrn_rate], val_feed)

                            # print the decode result
                            ori_labels = val_feeder.the_label(indexs_val)
                            
                            # 计算一个批次正确率
                            acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                             ignore_value=-1, isPrint=True)
                            acc_batch_total += acc

                        accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples # 求一轮的平均正确率

                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size) # 整一轮的当前平均损失值

                        # 输出训练最新信息
                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              "lastbatch_cost = {:f}, time = {:.3f},lr={:.8f}"
                        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                         batch_cost, time.time() - start_time, lr))

            
def infer(img_path, mode='infer'):
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = orcmodel.LSTMOCR(mode) # 生成模型类
    model.build_graph() # 建立模型

    total_steps = math.ceil(len(imgList) / FLAGS.batch_size) #获得最小批总批次

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer()) # 初始化

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100) # saver 获取变量
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt) # 恢复变量
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        true_expression=[]
        for curr_step in range(total_steps):

            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                # 准备预测数据
                true_expression.append(img.split('/')[-1].split('_')[-1].split('.')[0])
                im = np.array(Image.open(img).convert("L")).astype(np.float32)/255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

            imgs_input = np.asarray(imgs_input)
            seq_len_input = np.asarray(seq_len_input)
            seq_len_input = np.reshape(seq_len_input, [-1])

            feed = {model.inputs: imgs_input,
                    model.seq_len: seq_len_input}

            # 运行预测，获得解码结果
            dense_decoded = sess.run(model.dense_decoded, feed)
            
            # 解析对应的验证码
            for item in dense_decoded:
                expression = ''

                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)
        
        # 记录预测结果
        with open('./result.txt', 'w') as f:
            f.write("%-6s %-6s %-6s" % ("pred", "true", "is_right") + '\n')
            for code, true_code in zip(decoded_expression, true_expression):
                f.write("%-6s %-6s %-d" % (code, true_code, code==true_code) + '\n')

#main 一定要有参数，因为tf在调用的时候传入sys.argv
def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO) # 初始化日志等级
    tf.app.run() # 解析flags 并运行main函数
