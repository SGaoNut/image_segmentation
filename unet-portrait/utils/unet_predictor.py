import colorsys
import os
import sys
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from nets.unet import Unet as unet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

class UnetPredictor(object):

    def __init__(self, _defaults, **kwargs):
        self._defaults = _defaults
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        self.model = unet(self.model_image_size, self.num_classes)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self,image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh

    def detect_image(self, image, color="b", path_b=None):

        # 图像预处理
        image = image.convert('RGB')

        # 保持长宽比，进行resize
        img, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        img = np.asarray([np.array(img)/255])

        # 前向传播获得模型输出
        pr = self.model.predict(img)[0]

        # 后处理 获取人像mask
        pr = pr[:, :, 1].reshape([self.model_image_size[0], self.model_image_size[1]])
        # 去除灰边
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh),
             int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        # 证件照制作
        mask = pr  # array
        
                # 中值滤波
        def median_filter(mask):
            """
            :param mask:   mask, float, 值域[0, 1]
            :return:
            """
            pre_label_u8 = (mask * 255).astype(np.uint8)
            pre_label = cv2.medianBlur(pre_label_u8, 23)
            pre_label = (pre_label / 255).astype(np.float32)
            return pre_label

        mask = median_filter(mask)
        

        # 创建背景，白色、红色、蓝色
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  #
        h, w, c = img.shape
        if path_b:
            background = cv2.imread(path_b)
            background = cv2.resize(background, (w, h))
        else:
            background = np.zeros_like(img, dtype=np.uint8)
            if color == "b":
                background[:, :, 0] = 255
            elif color == "w":
                background[:] = 255
            elif color == "r":
                background[:, :, 2] = 255

        # 转换mask至BGR通道及指定size，alpha
        alpha_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha_bgr = cv2.resize(alpha_bgr, (w, h))

        # fusion
        result = np.uint8(img * alpha_bgr + background * (1 - alpha_bgr))

        return result


    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = np.asarray([np.array(img)/255])

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)

        t1 = time.time()
        for _ in range(test_interval):
            pr = self.model.predict(img)[0]
            pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
            image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time




