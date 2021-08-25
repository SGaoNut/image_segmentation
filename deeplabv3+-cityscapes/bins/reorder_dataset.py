# -*- coding: utf-8 -*-
"""
# @file name  : reorder_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2021-03-15
# @brief      : 重新组织cityscapes, mask要的是 labelIds.png
"""
import os
import shutil

def make_dir(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

out_dir = r"F:\tf2\DeepLabV3_Plus-Tensorflow2.0\cityscapes"
root_dir = r"G:\deep_learning_data\Cityspaces"

train_img = os.path.join(out_dir, "dataset", "train_images")
valid_img = os.path.join(out_dir, "dataset", "val_images")
train_msk = os.path.join(out_dir, "dataset", "train_masks")
valid_msk = os.path.join(out_dir, "dataset", "val_masks")
make_dir(train_img)
make_dir(valid_img)
make_dir(train_msk)
make_dir(valid_msk)


train_img_raw = os.path.join(root_dir, "images", "train")
valid_img_raw = os.path.join(root_dir, "images", "val")


def walk_png(root):
    img_lst = []
    for root, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".png"):
                path_img = os.path.join(root, f)
                img_lst.append(path_img)
    print(len(img_lst))
    return img_lst


train_img_lst = walk_png(train_img_raw)
valid_img_lst = walk_png(valid_img_raw)


def copy_png(img_lst_, out_img_dir, out_msk_dir):
    for idx, path in enumerate(img_lst_):
        set_name = path.split("\\")[-3]
        city_name = path.split("\\")[-2]
        img_name = path.split("\\")[-1]
        label_name_ele = img_name.split("_")[:-1]
        label_name_ele.append("gtFine_labelIds.png")
        label_name = "_".join(label_name_ele)

        img_dir = out_img_dir
        msk_dir = out_msk_dir

        # mask
        path_label_src = os.path.join(root_dir, "gtFine", set_name, city_name, label_name)
        path_label_dst = os.path.join(msk_dir, label_name)
        shutil.copy(path_label_src, path_label_dst)

        # img
        path_img_src = path
        path_img_dst = os.path.join(img_dir, img_name)
        shutil.copy(path_img_src, path_img_dst)
        print("{}/{}".format(idx, len(img_lst_)))

copy_png(train_img_lst, train_img, train_msk)
copy_png(valid_img_lst, valid_img, valid_msk)
print("done!")