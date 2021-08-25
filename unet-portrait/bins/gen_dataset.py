# -*- coding: utf-8 -*-
"""
# @file name  : gen_dataset.py
# @author     : https://github.com/TingsongYu
# @date       : 2020-06-01
# @brief      : portrait 2000， 生成voc数据集格式
"""
import os
import shutil
import random

random.seed(123)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_dir(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


if __name__ == '__main__':

    root_dir = r"G:\deep_learning_data\EG_dataset\dataset"
    out_dir = r"G:\deep_learning_data\EG_dataset\voc_format"
    # root_dir = os.path.join(BASE_DIR, "..", "..", "data", "dataset")
    # out_dir = os.path.join(BASE_DIR, "..", "..", "data", "dataset", "voc_format")
    train_per = 0.9

    # 读取所有数据集
    img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png") and not file.endswith("matte.png"):
                path_img = os.path.join(root, file)
                img_list.append(path_img)

    print("从{}文件夹下找到了{}张png图片".format(root_dir, len(img_list)))

    # 1、随机划分
    random.shuffle(img_list)

    split_point = int(len(img_list) * train_per)
    train_list = img_list[:split_point]
    valid_list = img_list[split_point:]
    train_msk_list = [i[:-4] + "_matte.png" for i in train_list]
    valid_msk_list = [i[:-4] + "_matte.png" for i in valid_list]
    print("按比例划分得到:{}训练，{}验证".format(len(train_list), len(valid_list)))

    # 2、复制图片
    def cp_batch(path_lst, dst_dir):
        for idx, p in enumerate(path_lst):
            src_path = p
            file_name = os.path.basename(p)
            base_dir = os.path.dirname(p)
            set_name = os.path.basename(base_dir)
            new_file_name = set_name + "_" + file_name
            dst_path = os.path.join(dst_dir, new_file_name)
            shutil.copy(src_path, dst_path)
            print("\r{}/{}".format(idx, len(path_lst)), end="", flush=True)


    out_img_dir = os.path.join(out_dir, "JPEGImages")
    out_msk_dir = os.path.join(out_dir, "SegmentationClass")
    make_dir(out_img_dir)
    make_dir(out_msk_dir)

    cp_batch(train_list + valid_list, out_img_dir)
    cp_batch(train_msk_list + valid_msk_list, out_msk_dir)
    print("按照VOC数据集目录形式，将数据存放于:{}\n标签存放于:{}".format(out_img_dir, out_msk_dir))

    # 3、生成txt
    txt_dir = os.path.join(out_dir, "ImageSets", "Segmentation")
    make_dir(txt_dir)
    path_train_txt = os.path.join(txt_dir, "train.txt")
    path_valid_txt = os.path.join(txt_dir, "val.txt")


    def gen_txt(path_lst, path_txt):
        "根据文件名生成txt文件"
        with open(path_txt, "w") as f:
            for path in path_lst:
                p = path
                file_name = os.path.basename(p)
                base_dir = os.path.dirname(p)
                set_name = os.path.basename(base_dir)
                new_file_name = set_name + "_" + file_name[:-4]
                f.writelines(str(new_file_name) + "\n")
        print("write done, path:{}".format(path_txt))


    gen_txt(train_list, path_train_txt)
    gen_txt(valid_list, path_valid_txt)


