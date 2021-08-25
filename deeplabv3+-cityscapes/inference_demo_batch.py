"""
批预测，以文件夹形式预测， validation数据，并保存png图片及生成 mp4视频
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from models.deeplab import DeepLabV3Plus
from tqdm import tqdm
import os
from glob import glob
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pipeline(image, fname='', folder=''):
    "接收单张图片，推理并保存分割结果图"
    global b
    image = cv2.resize(image, (w, h))
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)
    img_color = image.copy()
    for i in np.unique(y):
        if i in id_to_color:
            img_color[y == i] = id_to_color[i]
    cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
    return cv2.imwrite(os.path.join(folder, fname), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":

    # step0： 配置参数
    h, w = 800, 1600
    cls_num = 34        # 分类类别，用于创建模型
    alpha = 0.5         # 预测图与原图融合比例
    path_pkl = os.path.join(BASE_DIR, "data", "cityscapes_dict.pkl")
    path_model = os.path.join(BASE_DIR, "data", "weights_08-20_23-06-val088.h5")
    img_root_dir = r'G:\deep_learning_data\Cityspaces\images\val'
    out_dir = os.path.join(BASE_DIR, "output")
    id_to_color = read_pkl((path_pkl))

    # step1：模型定义
    model = DeepLabV3Plus(h, w, cls_num)
    model.load_weights(path_model)

    # step2：遍历文件夹里图片
    for img_dir in os.listdir(img_root_dir):
        out_sub_dir = os.path.join(out_dir, img_dir)
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)
        image_list = os.listdir(os.path.join(img_root_dir, img_dir))
        image_list.sort()
        print(f'{len(image_list)} frames found')
        # 遍历每张图片
        for i in tqdm(range(len(image_list))):
            try:
                path_img = os.path.join(img_root_dir, img_dir, image_list[i])
                img_array = img_to_array(load_img(path_img))
                segmap = pipeline(img_array, fname=f'{image_list[i]}', folder=out_sub_dir)
                if segmap == False:
                    break
            except Exception as e:
                print(str(e))
        clip = ImageSequenceClip(sorted(glob(out_sub_dir+r"\*")), fps=2, load_images=True)
        clip.write_videofile(os.path.join(out_sub_dir, r'{img_dir}.mp4'))
