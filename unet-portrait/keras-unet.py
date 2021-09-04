# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: keras-unet.py
@time: 2021/9/2 10:20 上午
"""
# import os
# import tensorflow as tf
#
# # 数据集创建
# BASE_DIR = os.getcwd()
# dataset_path = os.path.join(BASE_DIR, '..', 'data', 'dataset', 'voc_format')  # linux
# with open(os.path.join(dataset_path, 'ImageSets/Segmentation/train.txt'), 'r') as f:
#     train_lines = f.readlines()
# with open(os.path.join(dataset_path, 'ImageSets/Segmentation/test.txt'), 'r') as f:
#     test_lines = f.readlines()
#
# max_epoch = 100  # 总迭代轮
# BATCH_SIZE = 1  # 去修改下
# INPUTS_SIZE = [224, 224, 3]
# NUM_CLASSES = 2  # 模型输出通道数， 这包含背景类别数，本例中为 1+1=2   #
# LR = 1e-4
# DECAY_RATE = 0.95  # 指数衰减参数，每个epoch之后，学习率衰减率

# #  利用生成器创建dataset
# gen = Generator(BATCH_SIZE, train_lines, INPUTS_SIZE, NUM_CLASSES, dataset_path)
# gen = tf.data.Dataset.from_generator(partial(gen.generate, random_data=True), (tf.float32, tf.float32))
# gen = gen.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)
#
# gen_val = Generator(BATCH_SIZE, val_lines, INPUTS_SIZE, NUM_CLASSES, dataset_path)
# gen_val = tf.data.Dataset.from_generator(partial(gen_val.generate, random_data=False), (tf.float32, tf.float32))
# gen_val = gen_val.shuffle(buffer_size=BATCH_SIZE).prefetch(buffer_size=BATCH_SIZE)

from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(512, 512, 3),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')

print(model.summary())

history = model.fit_generator(...)

from keras_unet.utils import plot_segm_history

plot_segm_history(
    history, # required - keras training history object
    metrics=['iou', 'val_iou'], # optional - metrics names to plot
    losses=['loss', 'val_loss']) # optional - loss names to plot