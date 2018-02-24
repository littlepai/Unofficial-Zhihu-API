# Unofficial-Zhihu-API

## 简介
深度学习模型自动识别验证码，python爬虫库自动管理会话，通过简单易用的API，实现知乎数据的爬取
如果大家愿意向我推荐更多的功能请发送信息到txgyswh@163.com,我会添加更多的功能.
博客地址: [深度学习与爬虫实例教学](http://www.cnblogs.com/paiandlu/p/8462657.html)

## 获取代码
可点击 [下载](https://github.com/littlepai/Unofficial-Zhihu-API/archive/master.zip) 获取项目代码，并解压获得 **Unofficial-Zhihu-API** 文件夹

## 环境安装
建议使用anaconda3-5.0.1环境，可点击 [下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)安装anaconda，python版本**建议**使用python3.6
安装anaconda（不会安装请自行百度）之后，打开相应的命令行窗口，按照下列步骤安装python环境
> * conda create -n uf_zhihu python=3.6 jupyter
> * cd ../Unofficial-Zhihu-API/
> * activate uf_zhihu
> * python setup.py install

如果你的是windows用户，又没有装vs或者vs版本不对，可能会报如下错误
![](http://images.cnblogs.com/cnblogs_com/paiandlu/1165432/o_pip_tensorflow_error.bmp)
**解决方法：**用 conda install tensorflow==1.2.1 安装tensorflow完毕之后再来运行python setup.py install即可

安装完毕之后再python的site-packages目录下生成ufzh项目（之所以做成一整个包是为了方便以后整合到别的项目）

## 优先体验
### 验证码自动识别测试
```python
cd 到  Unofficial-Zhihu-API/train_workspace
ipython
import helper
helper.mark()
```

### 调用API获取数据
```python

ipython
from ufzh import zhihu
s=zhihu.Search()
s.relatedQidByKWord
java_qs=s.relatedQidByKWord("java") # 查询java相关的话题
java_q=java_qs.fetchone() # 先获取一部分数据
java_q #ipython 会打印详细信息

###################################
comments=s.commentByQid(java_q[0]['qid']) # 针对第一个问题，查找当前问题下的评论信息
comments_data=comments.fetchone() # 先获取一部分数据
comments_data #ipython 会打印详细信息，但是信息会很多，后面会补上字段说明

#####################################
comments_data[0]['author']["name"] # 第一位评论的用户的名字
url_token=comments_data[0]['author']["url_token"] # 该用户的令牌，zhihu为每一位用户唯一分配
followers=s.getFollowers(url_token) # 给定用户令牌，获取用户粉丝
followers_data=followers.fetchone() # 先获取部分数据
followers_data #打印每一位粉丝的详细信息，字段很多
```


## 扩展
主要用Unofficial-Zhihu-API进行自动登录，登陆之后获取一个包含登陆信息的requests会话进行 **自定义爬虫。**
```python
ipython
from ufzh import zhihu
s=zhihu.Zhihu()
#session是一个requests库的一个会话，使用方法请参考官方教程
session=s.getSession()

```


## 从零开始训练验证码识别
如果你需要学习或者体验 **如何在缺少知乎已标记验证码的情况下，训练验证码识别** ，请访问博客地址: [深度学习与爬虫实例教学](http://www.cnblogs.com/paiandlu/p/8462657.html)