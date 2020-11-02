import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
import os
import migrate
# from vgg16_original import vgg16_model_org
from vgg16 import vgg16_updated
from utils import custom_loss
from wsi_data_utils import patch_data_label_generator

if __name__ == '__main__':
    batch_size = 32

    # Load our model
    model_weights_path = '/infodev1/non-phi-data/junjiang/Autoencoder_plus/models/unsampled_model.01-1.1702.hdf5'
    model = vgg16_updated()
    model.load_weights(model_weights_path)
    print(model.summary())

    # Load our data
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"

    '''
    # testing data generator
    '''
    data_test_txt = os.path.join(patch_root, "testing.txt")
    test_gen = patch_data_label_generator(data_test_txt, batch_size=batch_size)

    test_img_fns = open(data_test_txt, "r").readlines()
    test_cnt = len(test_img_fns)

    for i in range(int(test_cnt/batch_size)):
        rgb_img_batch = next(test_gen)
        test_x = rgb_img_batch[0]
        test_y = rgb_img_batch[1]
        out = model.predict(test_x, batch_size=batch_size)

        # create heat-map

        test_i


    # prediction
    # prd = model.evaluate_generator(test_gen, int(test_cnt/batch_size))
    # input_fn = tf.estimator.inputs.numpy_input_fn(test_gen)
    # for single_prediction in model.predict(input_fn):
    #     predicted_class = single_prediction['class']
    #     probability = single_prediction['probability']
    # prd = model.predict(test_gen, batch_size=batch_size)


