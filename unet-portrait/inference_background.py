# -*- coding: utf-8 -*-
"""
# @file name  : pre.py
# @brief      : 推理代码，演示视频背景替换
"""
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
        "model_path": 'logs/model_weight_2021_08_20_13_50_54-336-0.95.h5',
        "model_image_size": (336, 336, 3),
        "NUM_CLASSES": 2,  # 21
    }
    path_back = os.path.join(BASE_DIR, "data", "bg1.jpg")

    # step1：创建预测模型类
    unet = UnetPredictor(_defaults)

    # step2：处理视频
    video_path = 0
    vid = cv2.VideoCapture(video_path + cv2.CAP_DSHOW)  # 0表示打开视频，1; cv2.CAP_DSHOW去除黑边

    while True:
        return_value, frame_bgr = vid.read()  # 读视频每一帧
        if not return_value:
            raise ValueError("No image!")
        # —————————————————————————————————————————————————————————————————— #
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        r_image = unet.detect_image(image, path_b=path_back)
        show_img = np.concatenate([frame_bgr, r_image], axis=1)
        # —————————————————————————————————————————————————————————————————— #
        cv2.imshow("result", show_img)
        # waitKey，参数是1，表示延时1ms；参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停
        # ord: 字符串转ASCII 数值
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()  # release()释放摄像头
    cv2.destroyAllWindows()  # 关闭所有图像窗口
