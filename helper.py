# coding: utf-8

import numpy as np
import os
import string
from pai_image import ImageCaptcha
from random import choice
from tqdm import tqdm

characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
# 随机分割训练集和测试集
def split_train_val(X, y, train_size):
    """
    将数据集切分为训练集和测试集

    参数:
        X: 一维 图片信息
        y: 一维 标记信息
        train_size: 训练集的大小
    返回:
        训练集图片，训练集label，测试集图片，测试集label
    """

    total_size = len(X)
    # shuffle data
    shuffle_indices = np.random.permutation(np.arange(total_size))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    # split training data
    train_indices = np.random.choice(total_size, train_size, replace=False)
    X_train = X[train_indices]
    y_train = y[train_indices]

    # split validation data
    val_indices = [i for i in range(total_size) if i not in train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val


# 给定目录，获取该目录下的图片路径列表
def load_img_path(images_path):
    tmp = os.listdir(images_path)
    #tmp.sort(key=lambda x: int(x.split('_')[0]))

    file_names = [os.path.join(images_path, s) for s in tmp]
    file_names = np.asarray(file_names)

    return file_names

# 生成模拟数据，模拟知乎的验证码，只有这样才能获得大量训练数据
def gen_simulated_img(images_path, num_imgs=10000, captcha_num=4, width=150, height=60, font_sizes=range(45,50)):
    '''
    默认只需要传入生成图片的路径即可,如果路径不存在则新建,如果存在则追加图片(用户可以自行删除已有的图片)
    '''
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    file_names = load_img_path(images_path)
    if file_names.size:
        file_names=[os.path.basename(file) for file in file_names]
        file_names.sort(key=lambda x: x.split('_')[0])
        nextid=int(os.path.basename(file_names[-1]).split('_')[0])+1
    else:
        nextid=0

    generator = ImageCaptcha(width=width, height=height, font_sizes=font_sizes)
    for i in tqdm(range(num_imgs), ncols=50):
        labels=''.join([choice(characters) for i in range(captcha_num)])
        generator.generate_image(labels).save("%s/%06d_%s.png" %(images_path, nextid, labels.lower()))
        nextid += 1