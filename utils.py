import os

import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar
from keras.optimizers import SGD


def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def load_data():
    # (num_samples, 256, 256, 3)
    num_samples = 8144
    train_split = 0.8
    batch_size = 16
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 256, 256, 3), dtype=np.float32)
    y_train = np.empty((num_train, 256, 256, 1), dtype=np.float32)
    x_valid = np.empty((num_valid, 256, 256, 3), dtype=np.float32)
    y_valid = np.empty((num_valid, 256, 256, 1), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if filename.startswith('data/train'):
                x_train[i_train, :, :, 0:3] = rgb_img / 255.
                x_train[i_train, :, :, 3] = np.random.uniform(0, 1, (256, 256))
                y_train[i_train, :, :, 0] = gray_img / 255.
                i_train += 1
            elif filename.startswith('data/valid'):
                x_valid[i_valid, :, :, 0:3] = rgb_img / 255.
                x_valid[i_valid, :, :, 3] = np.random.uniform(0, 1, (256, 256))
                y_valid[i_valid, :, :, 0] = gray_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar(i * 100 / num_samples)

    return x_train, y_train, x_valid, y_valid


def load_wsi_patches(patch_dir):
    # (num_samples, 256, 256, 3)
    num_samples = 8244
    train_split = 0.8
    batch_size = 16
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 256, 256, 3), dtype=np.float32)
    y_train = np.empty((num_train, 256, 256, 3), dtype=np.float32)
    x_valid = np.empty((num_valid, 256, 256, 3), dtype=np.float32)
    y_valid = np.empty((num_valid, 256, 256, 3), dtype=np.float32)

    train_img_fn_txt = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/train_img_fn.txt"
    test_img_fn_txt = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/test_img_fn.txt"
    train_fp = open(train_img_fn_txt, 'a')
    test_fp = open(test_img_fn_txt, 'a')

    i_train = i_valid = 0
    for root, dirs, files in os.walk(patch_dir, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if "144" not in filename and i_train < num_train:
                train_fp.write(filename+"\n")
                x_train[i_train, :, :, :] = rgb_img / 255.
                i_train += 1
            else:
                test_fp.write(filename + "\n")
                x_valid[i_valid, :, :, :] = rgb_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar(i * 100 / num_samples)

    y_train = np.copy(x_train)
    y_valid = np.copy(x_valid)

    train_fp.close()
    test_fp.close()

    return x_train, y_train, x_valid, y_valid



def do_compile(model):
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    # model.compile(optimizer='nadam', loss=custom_loss)
    model.compile(optimizer=sgd, loss=custom_loss)
    return model
