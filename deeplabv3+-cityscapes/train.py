import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from models.deeplab import DeepLabV3Plus
from utils.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print('TensorFlow', tf.__version__)
log_dir = create_logdir(BASE_DIR)

if __name__ == "__main__":

    # step0：配置参数
    root_dir = r"F:\tf2\DeepLabV3_Plus-Tensorflow2.0\cityscapes"  # must be reorder dirs!!!
    # root_dir = "/home/teacher/notebooks/cv_project_2days/object_detection/data/cityspaces_reorder"
    batch_size = 1  # 24
    H, W = 512, 512
    num_classes = 34
    MAX_EPOCH = 30  # 30-60
    lr_init = 1e-4
    decay_rate = 0.99
    debug_img_num = 0
    path_save_weight = os.path.join(log_dir, "top_weights.h5")

    # step1：数据集制作
    img_info = get_img_lst(root_dir, debug_img_num)
    image_list, mask_list, val_image_list, val_mask_list = img_info

    train_dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    train_dataset = train_dataset.shuffle(buffer_size=128).apply(
        tf.data.experimental.map_and_batch(map_func=load_data,  # 实现数据预处理功能
                                           batch_size=batch_size,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)  # 将整个数据重复多次，用于处理Epoch

    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_mask_list))
    val_dataset = val_dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_data,
                                           batch_size=batch_size,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    val_dataset = val_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    # step2：损失函数、优化器、学习率调整策略设置

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    epoch_size = len(image_list) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_init, decay_steps=epoch_size, decay_rate=decay_rate, staircase=True)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    # step3：模型搭建、编译
    model = DeepLabV3Plus(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    model.compile(loss=loss, optimizer=optimizer,  metrics=['accuracy'])

    # step4：训练模型
    tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
    mc = ModelCheckpoint(mode='min', filepath=path_save_weight, monitor='val_loss', save_best_only='True',
                         save_weights_only='True', verbose=1)
    callbacks = [mc, tb]

    model.fit(train_dataset,
              steps_per_epoch=len(image_list) // batch_size,
              epochs=MAX_EPOCH,
              validation_data=val_dataset,
              validation_steps=len(val_image_list) // batch_size,
              callbacks=callbacks)
