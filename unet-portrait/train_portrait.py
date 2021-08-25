import os
import sys
from functools import partial
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from nets.unet import Unet
from nets.unet_training import CE, Generator, LossHistory
from utils.metrics import f_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_train_step_fn():
    @tf.function
    def train_step(images, labels, net, optimizer, loss):
        with tf.GradientTape() as tape:
            # 计算loss
            prediction = net(images, training=True)
            # print("loss input: {} {}".format(labels.shape, prediction.shape))
            loss_value = loss(labels, prediction)

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        _f_score = f_score()(labels, prediction)
        # _f_score = Iou_score()(labels, prediction)
        return loss_value, _f_score
    return train_step


@tf.function
def val_step(images, labels, net, loss):
    # 计算loss
    prediction = net(images, training=False)
    loss_value = loss(labels, prediction)
    
    _f_score = f_score()(labels, prediction)
    return loss_value, _f_score


def fit_one_epoch(net, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, train_step):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, labels = batch[0], batch[1]
            labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

            loss_value, _f_score = train_step(images, labels, net, optimizer, loss)
            total_loss += loss_value.numpy()
            total_f_score += _f_score.numpy()

            pbar.set_postfix(**{'Total Loss'        : total_loss / (iteration + 1), 
                                'Total f_score'     : total_f_score / (iteration + 1),
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
        
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            images, labels = batch[0], batch[1]
            labels = tf.convert_to_tensor(labels)

            loss_value, _f_score = val_step(images, labels, net, loss)
            val_loss            += loss_value.numpy()
            val_f_score         += _f_score.numpy()

            pbar.set_postfix(**{'Val Loss'      : val_loss / (iteration + 1), 
                                'Val f_score'   : val_f_score / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss/(epoch_size+1), 'val_loss': val_loss/(epoch_size_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Finish Validation')
    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":

    # step0：参数配置
    dataset_path = r"G:\deep_learning_data\EG_dataset\voc_format"
    # dataset_path = os.path.join(BASE_DIR, "..", "data", "dataset", "voc_format")
    model_path = os.path.join(BASE_DIR, "data", "model_data", "unet_voc.h5")

    max_epoch = 100
    Batch_size = 1
    inputs_size = [224, 224, 3]
    num_classes = 2
    lr = 1e-4
    decay_rate = 0.95

    import datetime
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_history = LossHistory("logs/", time_str)
    log_dir = os.path.join(BASE_DIR, "logs", "loss_" + time_str)

    # step1：数据集创建
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    epoch_size = len(train_lines) // Batch_size  # 计算一个epoch有几个iteration
    epoch_size_val = len(val_lines) // Batch_size

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

    #  利用生成器创建dataset
    gen = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path)
    gen = tf.data.Dataset.from_generator(partial(gen.generate, random_data=True), (tf.float32, tf.float32))
    gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

    gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path)
    gen_val = tf.data.Dataset.from_generator(partial(gen_val.generate, random_data=False), (tf.float32, tf.float32))
    gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

    if epoch_size == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # step2： 创建model
    model = Unet(inputs_size, num_classes)  # 需要传入输入大小和最终输出通道数
    model.load_weights(model_path, by_name=True, skip_mismatch=True)  # 加载预训练模型

    # step3：创建loss及优化器
    loss = CE()
    lr_schedule = ExponentialDecay(initial_learning_rate=lr, decay_steps=epoch_size,
                                   decay_rate=decay_rate, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)

    # step4：迭代训练
    for epoch in range(max_epoch):
        fit_one_epoch(model, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, max_epoch, get_train_step_fn())
        path_model = os.path.join(log_dir, "model_weight_{}.h5".format(time_str))
        model.save_weights(path_model)