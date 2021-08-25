"""
推理单张图片
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from models.deeplab import DeepLabV3Plus
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":

    # step0：参数配置
    h, w = 800, 1600    # 图片输入大小 800, 1600
    cls_num = 34        # 分类类别，用于创建模型
    alpha = 0.5         # 预测图与原图融合比例

    path_img = os.path.join(BASE_DIR, "data", "cityscapes", "dataset", "val_images",
                            "frankfurt_000000_000294_leftImg8bit.png")
    path_pkl = os.path.join(BASE_DIR, "data", "cityscapes_dict.pkl")
    path_model = os.path.join(BASE_DIR, "data", "pretrained_model_git.h5")
    id_to_color = read_pkl((path_pkl))

    # step1：创建模型
    model = DeepLabV3Plus(h, w, cls_num)
    model.load_weights(path_model)

    # step2： 图片预处理
    img_array = img_to_array(load_img(path_img))  # 读取图片
    image = cv2.resize(img_array, (w, h))         # resize
    img_batch = np.expand_dims(image.copy(), axis=0)    # 增加batchsize 维度
    img_batch_norm = preprocess_input(img_batch)        # normalization

    # step3：推理
    output_batch = model.predict(img_batch_norm)    # 前向传播
    output = np.squeeze(output_batch)               # 剔除batchsize维度
    output_idx = np.argmax(output, axis=2)          # 分类概率向量 转为 分类类别

    # step4：分类类别矩阵 转为 RGB颜色矩阵
    img_color = image.copy()

    for i in np.unique(output_idx):     # 遍历输出标签类别，unique类似取集合，不会重复
        if i in id_to_color:
            img_color[output_idx == i] = id_to_color[i]  # 标签是i的位置， 赋值 id_to_color[i]

    # step5：融合图片，绘图
    cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
    img_color = img_color.astype(np.uint8)
    plt.imshow(img_color)
    plt.show()

    cv2.imshow("result", cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
