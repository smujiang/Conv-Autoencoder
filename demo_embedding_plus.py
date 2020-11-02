import os
import random
import tensorflow as tf
import cv2 as cv
import keras.backend as K
import numpy as np

from model import create_model, create_encoder
from sklearn.metrics import mean_squared_error
from wsi_data_utils import patch_data_label_generator

if __name__ == '__main__':
    img_rows, img_cols = 256, 256
    channel = 4
    batch_size = 16
    output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchAEClassification/PatchEmbeddings"
    # model_weights_path = 'models/model.24-0.0449.hdf5'
    model_weights_path = "models/model.61-0.0371.hdf5"
    # A: restore the encoder
    model = create_encoder()
    model_AE = create_model()
    # AE_layers = [l for l in model_AE.layers]
    for i in range(len(model.layers)):
        # old_layer = AE_layers[i]
        # new_layer = new_layers[i]
        model.layers[i].set_weights(model_AE.layers[i].get_weights())

    print(model.summary())
    embeddings = np.empty([0, 4, 4, 512], np.float)
    output_file = os.path.join(output_dir, "training_data.npz")
    if os.path.exists(output_file):
        raise Exception("output file already exist, get a new name for the file")
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"
    data_train_txt = os.path.join(patch_root, "training_shuffled.txt")
    samples = open(data_train_txt, 'r').readlines()
    for i in range(len(samples)):
        filename = samples[i].split(",")[1].strip()
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img / 255.0
        x_test[0, :, :, :] = rgb_img
        eb = model.predict(x_test)
        embeddings = np.vstack((embeddings, eb))
    np.savez(output_file, embeddings)

    K.clear_session()
