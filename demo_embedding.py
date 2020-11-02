import os
import random
import tensorflow as tf
import cv2 as cv
import keras.backend as K
import numpy as np

from model import create_model, create_encoder
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    img_rows, img_cols = 256, 256
    channel = 4

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
            # filename = os.path.join(test_path, image_name)

            print('Start processing image: {}'.format(filename))

            x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
            bgr_img = cv.imread(filename)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = rgb_img / 255.0
            x_test[0, :, :, :] = rgb_img
            embedding = model.predict(x_test)

            # with tf.Session() as sess:
            #     x_value = sess.run(embedding)
            #     print(x_value)
            #     write_to = os.path.join("/infodev1/non-phi-data/junjiang/Autoencoder_plus/data_log", "temp.npy")
            #     np.save(write_to, x_value, allow_pickle=False)

            # tf.io.write_file(write_to, embedding, "embedding")
            # print(out.shape)
            out = np.squeeze(embedding.flatten())
            # out = np.reshape(out, (img_rows, img_cols))
            out = out * 255.0

            img_fn = os.path.split(filename)[1]
            input_fn = 'images/' + img_fn.replace(".jpg", "_input.jpg")
            output_fn = 'images/' + img_fn.replace(".jpg", "_output.jpg")

            mse = (np.square(rgb_img - out)).mean(axis=None)
            error_list_str += input_fn + "," + output_fn + "," + str(mse)+"\n"

            out = out.astype(np.uint8)

            bgr_out = cv.cvtColor(out, cv.COLOR_RGB2BGR)

            cv.imwrite(input_fn, bgr_img)
            cv.imwrite(output_fn, bgr_out)

        fp = open(testing_eval_csv, 'w')
        fp.write(error_list_str)
        fp.close()

    K.clear_session()
