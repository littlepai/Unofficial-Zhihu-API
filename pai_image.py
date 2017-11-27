# coding: utf-8

import os
import random
from PIL import Image
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'font')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'euphemia.ttf')] # 默认字体文件

table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )



class ImageCaptcha():
    """
    用于生成验证码
    验证码的每一个字母的字体和字体大小是fonts，font_sizes里面随机抽取的组合

    实例化参数：
        width：整数，验证码图片的长度
        height：整数，验证码图片的宽度
        fonts：可迭代对象，如list，list里面元素是字体文件
        font_sizes：可迭代对象，如list，list里面元素是字体大小

    """
    def __init__(self, width=150, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        """
        属性方法
        组合所有字体和字体大小
        """
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    def create_captcha_image(self, chars, color, background):
        """
        生成验证码图片
        
        参数：
            chars：text，验证码的字母
            color：前景颜色，也就是字体颜色
            background：背景颜色

        返回：
            返回一张验证码图片
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            """
            生成单个字母的图片
            """
            font = random.choice(self.truefonts) # 随机选择一种字体和字体大小
            w, h = draw.textsize(c, font=font) # 获得在改字体和大字体小下需要多大画布

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy)) # 创建四通道空图片
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # 旋转
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.NEAREST, expand=1)
            
            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images]) # 获得所有字母图片的宽之和，也就是验证码最小宽度

        width = max(text_width, self._width) # 获得验证码实际可用宽度
        image = image.resize((width, self._height)) # 塑型成指定大小

        average = int(text_width / len(chars)) # 单字母图片平均长度
        rand = int(0.4 * average) #字母之间融合区大小
        offset = random.randint(0, (width-text_width)) # 随机字母的水平偏置

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            h_offset=int((self._height - h) / 2) # 字母的垂直位置
            try:
                dh = int(random.randint(-h_offset, h_offset) * 0.5) # 随机字母的垂直偏置
            except ValueError:
                dh=0
            image.paste(im, (offset, h_offset+dh), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """
        给定字符串，生成验证码

        参数：
            chars: text，验证码字符串
        """
        background = random_color(0, 0) #黑底
        color = random_color(255, 255, 255) #前景（字体）白色并且完全不透明
        im = self.create_captcha_image(chars, color, background)
        im = im.point(lambda i: 255-i)
        return im


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)
