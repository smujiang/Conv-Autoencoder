from keras.models import Model
from model import create_model, create_encoder
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import numpy as np
from vgg16 import vgg16_updated

img_rows, img_cols = 256, 256
output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/VGG16_activation"

vgg_model = vgg16_updated()
# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in vgg_model.layers]
# Creates a model that will return these outputs, given the model input
activation_model = Model(inputs=vgg_model.input, outputs=layer_outputs)



test_type_list = ["Borderline", "HighGrade"]
for test_type in test_type_list:
    if test_type is "HighGrade":
        test_path = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256/OCMC-001"
        img_fn_list = os.listdir(test_path)
        test_images = []
        for i_f in img_fn_list:
            test_images.append(os.path.join(test_path, i_f))
        testing_eval_csv = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/MSE_eval_highgrade.csv"
    elif test_type is "Borderline":
        testing_img_fn = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/test_img_fn.txt"
        test_images = open(testing_img_fn, 'r').readlines()

        testing_eval_csv = "/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log/MSE_eval_borderline.csv"
    else:
        raise Exception("undefined evaluation type")

    samples = random.sample(test_images, 100)
    error_list_str = "img_input,img_output,error\n"
    for i in range(len(samples)):
        filename = samples[i].strip()
        fn = os.path.split(filename)[1]
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img / 255.0
        x_test[0, :, :, :] = rgb_img

        # Returns a list of five Numpy arrays: one array per layer activation
        activations = activation_model.predict(x_test)
        plt.imshow(rgb_img)
        plt.axis("off")

        img_out_path = os.path.join(output_dir, fn[0:-4])
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)
        plt.savefig(os.path.join(output_dir, fn[0:-4], fn))
        for idx, act_layer in enumerate(activations):
            for l in range(act_layer.shape[3]):
                plt.imshow(act_layer[0, :, :, l]*255, 'gray')
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, fn[0:-4], "layer_" + str(idx) + "_channel_" + str(l) + "_.jpg"))








