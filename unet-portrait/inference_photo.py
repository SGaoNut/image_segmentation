# -*- coding: utf-8 -*-
"""
# @file name  : pre.py
# @brief      : 推理代码，演示证件照制作过程
"""
import time
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.unet_predictor import UnetPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    _defaults = {
        "model_path": 'logs/model_weight_2021_08_19_11_05_59-float-label.h5',
        "model_image_size": (224, 224, 3),
        "num_classes": 2,  # 21
    }
    background_color = "b"  # b:蓝色， r：红色， w：白色
    # path_img = os.path.join(BASE_DIR, "data", "00079.png")
    path_img = os.path.join(BASE_DIR, "data", "00085.png")
    # path_img = os.path.join(BASE_DIR, "data", "00087.png")

    # step1：创建预测模型类
    unet = UnetPredictor(_defaults)

    # step2：处理图片
    image = Image.open(path_img)
    r_image = unet.detect_image(image, background_color)

    # step3：显示图片
    img_bgr = cv2.imread(path_img)
    show_img = np.concatenate([img_bgr, r_image], axis=1)
    # cv2.imwrite("t.png", show_img)

    cv2.imshow("result", show_img)
    cv2.waitKey()

