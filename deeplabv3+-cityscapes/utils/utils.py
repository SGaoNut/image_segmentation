import pickle
import cv2
import numpy as np
import os
import tensorflow as tf
from glob import glob


def read_pkl(path_pkl):
    with open(path_pkl, 'rb') as f:
        id_to_color = pickle.load(f)['color_map']
    return id_to_color


def create_logdir(BASE_DIR):
    from datetime import datetime
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR,  "logs", time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("make dir：", log_dir)
    return log_dir


def get_image(image_path, img_height=800, img_width=1600, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
    return img


def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask


def load_data(image_path, mask_path, H=512, W=512):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    return image, mask


def get_img_lst(root_dir, debug_img_num):
    train_img = os.path.join(root_dir, "dataset", "train_images")
    valid_img = os.path.join(root_dir, "dataset", "val_images")
    train_msk = os.path.join(root_dir, "dataset", "train_masks")
    valid_msk = os.path.join(root_dir, "dataset", "val_masks")

    image_list = sorted(glob(train_img + '/*'))
    mask_list = sorted(glob(train_msk + '/*'))
    val_image_list = sorted(glob(valid_img + '/*'))
    val_mask_list = sorted(glob(valid_msk + '/*'))

    if debug_img_num:
        image_list = image_list[:debug_img_num]
        mask_list = mask_list[:debug_img_num]
        val_image_list = val_image_list[:debug_img_num]
        val_mask_list = val_mask_list[:debug_img_num]

    print('Found', len(image_list), 'training images')
    print('Found', len(val_image_list), 'validation images')
    return image_list, mask_list, val_image_list, val_mask_list
