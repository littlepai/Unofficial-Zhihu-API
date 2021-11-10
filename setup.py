#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: littlepai
# Mail: txgyswh@163.com
#############################################


from setuptools import setup, find_packages

setup(
    name = "ufzh",
    version = "1.0.1",
    keywords = ("pip", "ufzh"),
    description = "深度学习实现知乎验证码识别",
    long_description = "深度学习实现知乎验证码识别",
    license = "MIT Licence",

    author = "littlepai",
    author_email = "txgyswh@163.com",

    packages = find_packages(),
    include_package_data = True,
    data_files=[('ufzh/checkpoint', ['ufzh/checkpoint/checkpoint', 'ufzh/checkpoint/ocr-model-22001.data-00000-of-00001', 'ufzh/checkpoint/ocr-model-22001.index','ufzh/checkpoint/ocr-model-22001.meta'])],
    platforms = "any",
    install_requires = ['requests', "bs4", 'tensorflow == 2.5.2 ', 'pillow', 'numpy', 'tqdm']
)
