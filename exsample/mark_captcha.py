import sys
import os
sys.path.append(os.path.abspath("../"))
import zhihu
from PIL import Image
from io import BytesIO
import shutil
from tqdm import tqdm


client=zhihu.Zhihu()

def mark(num=50):
    dirname="mark"
    if os.path.exists(dirname):
        shutil.rmtree('mark')
    os.mkdir(dirname)
    captchaURL = r"https://www.zhihu.com/captcha.gif?type=login"
    print("\n\n"+"*"*50)
    print("开始测试验证码识别功能")
    for i in tqdm(range(num), ncols=50):
        img=Image.open(BytesIO(client.open(captchaURL).content))
        expression=client.recgImg(img)
        img.save(os.path.join(dirname, expression+".png"))
    print("\n在当前目录下有mark文件夹，里面是刚刚识别的结果，如果有兴趣，可以统计正确率\n")

