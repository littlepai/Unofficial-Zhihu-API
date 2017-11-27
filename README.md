# Unofficial-Zhihu-API

## 简介
深度学习模型自动识别验证码，python爬虫库自动管理会话，通过简单易用的API，实现知乎数据的爬取
如果大家愿意向我推荐更多的功能请发送信息到txgyswh@163.com,我会添加更多的功能.

## 获取
可点击 [下载](https://github.com/littlepai/Unofficial-Zhihu-API/archive/master.zip) 获取项目代码，并解压获得Unofficial-Zhihu-API文件夹

确认python环境已经具备如下的库

|库名|库名|库名|库名|
|---|---|
|requests|bs4|webbrowser|tensorflow|
|pillow|numpy|tqdm|json|


## 优先体验
### 验证码自动识别测试
```python
cd ../Unofficial-Zhihu-API/
ipython
import exsample.mark_captcha as exm
exm.mark()
```

### 调用API获取数据
```python
cd ../Unofficial-Zhihu-API/
ipython
import zhihu
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