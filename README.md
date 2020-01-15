# Unofficial-Zhihu-API

[Pytorch版本](https://github.com/littlepai/tl_ocr)

## 简介
深度学习模型自动识别验证码，python爬虫库自动管理会话，通过简单易用的API，实现知乎数据的爬取
如果大家愿意向我推荐更多的功能请发送信息到txgyswh@163.com,我会添加更多的功能.
博客地址: [深度学习与爬虫实例教学](http://www.cnblogs.com/paiandlu/p/8462657.html)

## 获取代码
可点击 [下载](https://github.com/littlepai/Unofficial-Zhihu-API/archive/master.zip) 获取项目代码，并解压获得 **Unofficial-Zhihu-API** 文件夹

## 环境安装
建议使用anaconda3-5.0.1环境，可点击 [下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)安装anaconda，python版本**建议**使用python3.6
安装anaconda（不会安装请自行百度）之后，打开相应的命令行窗口，按照下列步骤安装python环境，linux 用户把下面的activate uf_zhihu换成source activate uf_zhihu
> * conda create -n uf_zhihu python=3.6 jupyter
> * cd ../Unofficial-Zhihu-API/
> * activate uf_zhihu
> * conda install tensorflow==1.14.0
> * python setup.py install

安装完毕之后再python的site-packages目录下生成ufzh项目（之所以做成一整个包是为了方便以后整合到别的项目）
说明：之所以在python setup.py install 之前执行conda install tensorflow==1.14.0是因为setup.py 安装的时候不能自动安装，不知道为啥
再说明：这个项目训练的时候其实使用tensorflow==1.2.1版本训练的，现在都2.0了，发现安装1.14.0的也行

## 优先体验
### 验证码自动识别测试
```python
cd 到  Unofficial-Zhihu-API/train_workspace
ipython
import helper
helper.mark()
```

## 爬虫部分将不可用，项目将作为学习验证码识别练习项目（缺数据的情况下）
本来要删了，因为这个项目当初是想用来做爬虫的，但是开发了验证码自动识别之后，就没之后了

## 从零开始训练验证码识别
如果你需要学习或者体验 **如何在缺少知乎已标记验证码的情况下，训练验证码识别** ，请访问博客地址: [深度学习与爬虫实例教学](http://www.cnblogs.com/paiandlu/p/8462657.html)
