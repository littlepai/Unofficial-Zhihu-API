# coding: utf-8

import requests
import time
import json
import os
import sys
from bs4 import BeautifulSoup as BS
import urllib.parse
import webbrowser

from io import BytesIO
from ufzh import utils
from ufzh import orcmodel
import tensorflow as tf
from PIL import Image
import numpy as np

try:
    type (eval('model'))
except:
    model = orcmodel.LSTMOCR('infer')
    model.build_graph()

config = tf.ConfigProto(allow_soft_placement=True)
checkpoint_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")

class ErrorNotFoundData(Exception):
    pass
class ErrorZhihuLimit(Exception):
    pass

# 重试装饰器
def connretry(trytimes=5, repdelay=2):
    """
    用于连接重试，因为知乎会有可能会检测到爬虫
    被ban之后可以重试
    
    参数：
        trytimes    重试的次数,默认5次
        repdelay    每次重试的间隔，默认2秒
    
    用法：
        @connretry(trytimes=10, repdelay=5):
        def my_func():
            pass
    """
    def __connretry(func):
        def __wrap(*args, **kargs):
            ret=None
            for i in range(trytimes):
                try:
                    ret=func(*args, **kargs)
                except json.JSONDecodeError:
                    print('内部函数涉及JSON解析，但是被解析的并不满足JSON格式')
                    return None
                except:
                    print('conn retry %d times' % (i))
                    raise
                    time.sleep(repdelay)
                else:
                    break
            return ret
        return __wrap
    return __connretry
                
        

class Fetch():
    """
    Fetch类主要是为了方便控制请求时间，避免被服务器ban掉
    主要方法：
        fetchone 一次取出一部分， 如果数据源头已空则不能再取，否则抛出ErrorNotFoundData异常，
                 如果要调用fetchone，请考虑和捕获ErrorNotFoundData异常
        
        fetchall 一次性取出全部，知道数据源头取空
        
    """
    def __init__(self, parse, url):
        self.__parse=parse
        self.__next=url
        self.__is_all=False
    
    def fetchone(self):
        """
        一次取出一部分数据，由于数据已经筛选过，所以取出
        的数据个数不一定都相等，数据取空之后还取，回抛出
        ErrorNotFoundData错误，如果不想捕获数据可以考虑
        fetchall()
        
        参数：
            无
        
        返回：
            list
        """
        if self.__is_all:
            raise ErrorNotFoundData('数据源头已空')
        data, self.__next = self.__parse(self.__next)
        if not self.__next:
            self.__is_all=True
        return data

    def fetchall(self):
        """
        一次性取出所有数据，数据源头如果空了会返回空列表
        
        参数：
            无
            
        返回：
            list
        """
        data=[]
        while self.__is_all==False:
            data.extend(self.fetchone())
        return data
        
        

class Zhihu():
    """
    Zhihu类实现自动登陆功能，包括自动识别验证码
    自动识别验证码是通过深度学习学到模型来识别
    
    实例化：
        client = Zhihu()或者client = Zhihu(username, password)
        可以使用邮箱登陆也可以使用注册的手机登陆
    
    相关功能介绍：
        testRecgImg 方法可以在线测试验证码功能是否好用
        recgImg 方法也是识别验证码，但是没有testRecgImg简单
        getLoginToken 方法可以获取当前登陆用户的令牌 user_token
        open 方法可以用来打开一些网址
    """

    # 网址参数是账号类型
    TYPE_PHONE_NUM = "phone_num"
    TYPE_EMAIL = "email"
    loginURL = r"https://www.zhihu.com/login/{0}"
    homeURL = r"https://www.zhihu.com"
    captchaURL = r"https://www.zhihu.com/captcha.gif?type=login"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.zhihu.com",
        "Upgrade-Insecure-Requests": "1",}

    cookieFile = os.path.join(sys.path[0], "cookie")

    def __init__(self, username=None, password=None):
        if sys.path[0]: os.chdir(sys.path[0])  # 设置脚本所在目录为当前工作目录

        # 恢复权重
        self.__sess = self.__restoreSess(checkpoint_dir)

        # 维护一个会话
        self.__session = requests.Session()
        self.__session.headers = self.headers #为了伪装，设置headers

        self.login(username, password)

    # 恢复权重
    def __restoreSess(self, checkpoint=checkpoint_dir):
        sess=tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(checkpoint)
        if ckpt:
            #回复权限，这里连 global_step 也会被加载进来
            saver.restore(sess, ckpt)
            # print('restore from the checkpoint{0}'.format(ckpt))
            print('已加载checkpoint{0}'.format(ckpt))
        else:
            print('警告：未加载任何chechpoint')
            print('如果这不是你预期中的，请确保以下目录存在可用的checkpoint:\n{0}'.format(checkpoint_dir))
        return sess

    # 登录
    def login(self, username=None, password=None):
        """
        验证码错误返回
        {'errcode': 1991829, 'r': 1, 'data': {'captcha': '请提交正确的验证码 :('}, 'msg': '请提交正确的验证码 :('}
        登录成功返回
        {'r': 0, 'msg': '登陆成功'}
        """
        
        print("\n"*5+"=" * 50)
        self.__loadCookie()
        if self.getLoginToken():
            print('Cookies认证成功')
            self.__is_login=True
            return True
        else:
            print('Cookies 认证失败，将尝试通过账号密码验证登陆')
        
        self.__username = username if username else input("请输入知乎用户名：")
        self.__password = password if password else input("请输入知乎密码：")
        self.__loginURL = self.loginURL.format(self.__getUsernameType())
        # 随便开个网页，获取登陆所需的_xsrf
        html = self.open(self.homeURL).text
        soup = BS(html, "html.parser")
        _xsrf = soup.find("input", {"name": "_xsrf"})

        for i in range(10):
            # 下载验证码图片
            img=Image.open(BytesIO(self.open(self.captchaURL).content))

            captcha = self.recgImg(img)
            # 发送POST请求
            data = {
                "_xsrf": _xsrf,
                "password": self.__password,
                "remember_me": "y",
                self.__getUsernameType(): self.__username,
                "captcha": captcha
            }
            resp_data = self.__session.post(self.__loginURL, data=data).json()
            print("=" * 50)
            if resp_data["r"] == 0:
                print("登录成功")
                self.__is_login = True
                print("=" * 50)
                print("在工作目录下生成cookie文件：%s, 下次登陆优先使用cookie登陆，\n如果不想优先使用cookie登陆可删除cookie文件" % self.cookieFile)
                self.__saveCookie()
                self.__is_login=True
                return True
            else:
                print("登录失败")
                self.__is_login = False
                if resp_data.get('data') and resp_data.get('data').get("name") == 'ERR_VERIFY_CAPTCHA_TOO_QUICK':
                    print('提交过快导致服务器怀疑是爬虫而拒接访问，请稍后重试或者重新打开另一个会话重试')
                    print('10秒之后重试')
                    time.sleep(10)
                elif resp_data.get('errcode') == 100005:
                    print(resp_data.get("msg"))
                    self.__is_login=False
                    return False
                else:
                    print(resp_data.get("msg"))
                    print('3秒之后重试')
                    time.sleep(3)
                    
        self.__is_login=False
        return False
    
    def logOut(self):
        self.open('https://www.zhihu.com/logout')
        self.__is_login=False
        
    def testRecgImg(self, num=1):
        """
        可以在线测试验证码识别功能
        参数：
            num 整数 是希望测试多少张，默认测试一张
            
        示例：
            client.testRecgImg()
            
            client.testRecgImg(5)
        """
        for i in range(num):
            img=Image.open(BytesIO(self.open(self.captchaURL).content))
            img.show()
            captcha=self.recgImg(img)
            print('识别结果：%s' %(captcha))
    
    def recgImg(self, img):
        """
        可以在线测试验证码识别功能
        参数：
            img 一个 (60, 150) 的图片
        """
        im = np.array(img.convert("L")).astype(np.float32)/255.
        im = np.reshape(im, [60, 150, 1])
        inp=np.array([im])
        seq_len_input=np.array([np.array([64 for _ in inp], dtype=np.int64)])
        #seq_len_input = np.asarray(seq_len_input)
        seq_len_input = np.reshape(seq_len_input, [-1])
        imgs_input = np.asarray([im])
        feed = {model.inputs: imgs_input,model.seq_len: seq_len_input}
        dense_decoded_code = self.__sess.run(model.dense_decoded, feed)
        expression = ''
        for i in dense_decoded_code[0]:
            if i == -1:
                expression += ''
            else:
                expression += utils.decode_maps[i]
        return expression
    
    # 返回登陆用户的Token，也可以当作判断是否登陆的方法
    def getLoginToken(self):
        """
        用户登陆之后
        可以调用该方法获取登陆用户的用户令牌 user_token
        
        参数：
            无
        """
        try:
            soup = BS(self.open(r"https://www.zhihu.com/").text, "html.parser")
            userTokens=soup.find("div", attrs={"data-zop-usertoken":True}).get("data-zop-usertoken")
            userToken=json.loads(userTokens)

            return userToken["urlToken"]
        except:
            return None
    
    def __getUsernameType(self):
        """判断用户名类型
        经测试，网页的判断规则是纯数字为phone_num，其他为email
        """
        if self.__username.isdigit():
            return self.TYPE_PHONE_NUM
        return self.TYPE_EMAIL

    def __saveCookie(self):
        """cookies 序列化到文件
        即把dict对象转化成字符串保存
        """
        with open(self.cookieFile, "w") as output:
            cookies = self.__session.cookies.get_dict()
            json.dump(cookies, output)

    def __loadCookie(self):
        """
        用密码登陆之后会在工作目录load下一个cookie文件
        如果下次登陆的时候已存在cookie文件则会使用cookie登陆
        读取cookie文件，返回反序列化后的dict对象，没有则返回None
        """
        if os.path.exists(self.cookieFile):
            with open(self.cookieFile, "r") as f:
                cookie = json.load(f)
                self.__session.cookies.update(cookie)
                return cookie
        return None

    def open(self, url, delay=0, timeout=10):
        """
        打开网页，返回Response对象
        参数：
            delay 整数 延迟多少秒打开，默认不延迟
            timeout 设置最大等待时间，超过该设定时间报超时错误，默认10秒
        """
        if delay:
            time.sleep(delay)
        return self.__session.get(url, timeout=timeout)

    def getSession(self):
        """
        获取当前实例维护的会话
        参数：
            无
        """
        return self.__session

class Search(Zhihu):
    """
    主要的api类
    
    相关方法：
        relatedQidByKWord 可以根据某些关键字搜索相关问题
        
        commentByQid 根据问题的id好，搜索该问题相关的评论
        
        getFollowing 给定用户令牌，获取该用户所关注的人（主动关注别人）的相关信息
        
        getFollowers 给定用户令牌，获取该用户的粉丝的相关信息
        
        如果你有其他想添加的功能，发送想实现的功能，我会升级更新，实现更多功能
        邮箱txgyswh@163.com
        
        
    """
    def __init__(self, username=None, password=None, searchurl='https://www.zhihu.com/r/search?q=%s&correction=1&type=content&offset=0'):
        super(Search, self).__init__(username, password)
        self.__searchurl=searchurl
        self.__commentByQidUrl='https://www.zhihu.com/api/v4/questions/%s/answers?include=data[*].is_normal,admin_closed_comment,reward_info,is_collapsed,annotation_action,annotation_detail,collapse_reason,is_sticky,collapsed_by,suggest_edit,comment_count,can_comment,content,editable_content,voteup_count,reshipment_settings,comment_permission,created_time,updated_time,review_info,question,excerpt,relationship.is_authorized,is_author,voting,is_thanked,is_nothelp,upvoted_followees;data[*].mark_infos[*].url;data[*].author.follower_count,badge[?(type=best_answerer)].topics&offset=1&limit=20&sort_by=default'

        self.openner=self.getSession()
        self.nexturl=None

    def relatedQidByKWord(self, word):
        """
        根据关键字搜索问题
        参数:
            word: 搜索关键字
        返回 Fetch 实例
        """

        starturl=self.__searchurl % word
        
        @connretry(trytimes=5, repdelay=2)
        def parse(url):
            respdata=self.open(url).json()
            data=[]
            nexturl=respdata['paging']['next']#self.__searchurl
            if nexturl: nexturl=urllib.parse.urljoin(url, nexturl)
            htmls=respdata['htmls']
            for html in htmls:
                try:
                    soup=BS(html, 'lxml')
                    qtag=soup.find('a', class_="js-title-link")
                    if qtag:
                        qurlinfo=qtag.get('href').split('/')
                    if qurlinfo and qurlinfo[-2]=='question':
                        qtext=qtag.text
                        data.append({'qid':qurlinfo[-1], 'text':qtext})
                except AttributeError as e:
                    continue
            return data, nexturl
        return Fetch(parse, starturl)
    
    def commentByQid(self, Qid):
        """
        根据问题的ID搜索问题的详细信息
        一般结合relatedQid得到的问题id，然后根据问题ID来获取详细信息
        参数:
            Qid: 一个整数，问题ID
        返回 Fetch 实例
        """
        commentByQidUrl = self.__commentByQidUrl % Qid
        
        @connretry(trytimes=5, repdelay=2)
        def parse(url):
            respdata = self.open(url).json()
                
            nexturl=None
            try:
                if not respdata['paging']['is_end']: nexturl = respdata['paging']['next']
            except:
                if 'error' in respdata:
                    print('系统检测到您的帐号或IP存在异常流量，请输入以下字符用于确认这些请求不是自动程序发出的')
                    webbrowser.open(respdata['error']['redirect'])
                    input('请在浏览器上打开知乎解除限制，然后会车')
                    nexturl=url
                return [], nexturl
            if nexturl == url: print('inurl = outurl')
            return respdata['data'], nexturl
        
        return Fetch(parse, commentByQidUrl)
    
    def getFollowing(self, user_token):
        """
        给定用户的token
        搜索该用户的主动关注列表
        """
        followingUrl=r'https://www.zhihu.com/api/v4/members/%s/followees?include=data[*].answer_count,articles_count,gender,follower_count,is_followed,is_following,badge[?(type=best_answerer)].topics&offset=0&limit=20' % user_token
        
        def parse(url):
            respdata=self.open(urllib.parse.urljoin(followingUrl, url)).json()
            nexturl=None
            try:
                if not respdata['paging']['is_end']: nexturl = respdata['paging']['next']
            except:
                if 'error' in respdata:
                    print('系统检测到您的帐号或IP存在异常流量，请输入以下字符用于确认这些请求不是自动程序发出的')
                    webbrowser.open(respdata['error']['redirect'])
                    input('请在浏览器上打开知乎解除限制，然后会车')
                    nexturl=url
                return [], nexturl
            return respdata['data'], nexturl
        return Fetch(parse, followingUrl)

    def getFollowers(self, user_token):
        """
        给定用户的token
        搜索该用户的被动关注（粉丝）列表
        """
        followerUrl=r'https://www.zhihu.com/api/v4/members/%s/followers?include=data[*].answer_count,articles_count,gender,follower_count,is_followed,is_following,badge[?(type=best_answerer)].topics&offset=0&limit=20' % user_token

        def parse(url):
            nexturl=None
            respdata=self.open(urllib.parse.urljoin(followerUrl, url)).json()
            try:
                if not respdata['paging']['is_end']: nexturl = respdata['paging']['next']
            except:
                if 'error' in respdata:
                    print('系统检测到您的帐号或IP存在异常流量，请输入以下字符用于确认这些请求不是自动程序发出的')
                    webbrowser.open(respdata['error']['redirect'])
                    input('请在浏览器上打开知乎解除限制，然后会车')
                    nexturl=url
                return [], nexturl
            return respdata['data'], nexturl

        return Fetch(parse, followerUrl)
