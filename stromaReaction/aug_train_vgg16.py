import random
import os
import staintools
from PIL import Image

header = "Fibrosis,Cellularity,Orientation,img_fn\n"
# csv_file = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation_pro/all_samples.csv"
# shuffled_training_csv_file = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"
# shuffled_validation_csv_file = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation_pro/training_five_cases.csv"

csv_file = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/all_samples.csv"
shuffled_training_csv_file = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/training_five_cases.csv"
shuffled_validation_csv_file = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/PatchSampling/validation_five_cases.csv"

lines = open(csv_file).readlines()[1:]
training_cnt = int(len(lines)*0.8)
validate_cnt = int(len(lines)*0.2)

random.shuffle(lines)
training_lines = lines[0:training_cnt]
validate_lines = lines[training_cnt:]

fp = open(shuffled_training_csv_file, 'w')
fp.write(header)
fp.writelines(training_lines)
fp.close()

fp = open(shuffled_validation_csv_file, 'w')
fp.write(header)
fp.writelines(validate_lines)
fp.close()

'''
###################### create data generator directly from image folder############################
'''
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import time
import tensorflow.keras.backend as K
import pandas as pd
import tensorflow as tf
import numpy as np
import staintools

tf.compat.v1.enable_eager_execution()
# if tf.executing_eagerly():
#     print("executing_eagerly")

IMG_SHAPE = (256, 256, 3)
num_classes = 3   # score from 1 to 3

all_label_list = ["Fibrosis", "Cellularity", "Orientation"]
names = header.strip().split(",")
train_df = pd.read_csv(shuffled_training_csv_file, header=0)
train_file_list = pd.Series.get(train_df, "img_fn").tolist()
# train_Fibrosis_scores_list = pd.Series.get(train_df, "Fibrosis").tolist()
# train_Cellularity_scores_list = pd.Series.get(train_df, "Cellularity").tolist()
# train_Orientation_scores_list = pd.Series.get(train_df, "Orientation").tolist()

val_df = pd.read_csv(shuffled_validation_csv_file, header=0)
val_file_list = pd.Series.get(val_df, "img_fn").tolist()
# val_Fibrosis_scores_list = pd.Series.get(val_df, "Fibrosis").tolist()
# val_Cellularity_scores_list = pd.Series.get(val_df, "Cellularity").tolist()
# val_Orientation_scores_list = pd.Series.get(val_df, "Orientation").tolist()

class WSI_data_generator():

    def __init__(self, file_list, label_list, batch_size, augment_opt):
        self.flip_up_down = False
        self.flip_left_right = False
        self.stain_variations = False
        self.color_variations = False
        self.batch_size = batch_size
        self.stain_augmentor = None
        #TODO: input data is a tf-record file
        self.file_list = file_list
        self.label_list = label_list
        if "flip_up_down" in augment_opt:
            self.flip_up_down = True
        if "flip_left_right" in augment_opt:
            self.flip_left_right = True
        if "stain_variations" in augment_opt:
            self.stain_variations = True
            self.stain_augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
        if "color_variations" in augment_opt:
            self.color_variations = True

# Too time consuming, 4 epochs in 18 hours
    '''
    2021-01-04 09:14:46.700131: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 536870912 exceeds 10% of free system memory.
    280/280 [==============================] - 16217s 58s/step - loss: 0.6063 - accuracy: 0.4949 - val_loss: 0.5997 - val_accuracy: 0.5063
    Epoch 3/200
    280/280 [==============================] - ETA: 0s - loss: 0.6058 - accuracy: 0.4955 
    Epoch 00003: val_loss improved from 0.59967 to 0.59848, saving model to /infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/model/Fibrosis/Fibrosis_03-0.5985.hdf5
    280/280 [==============================] - 16213s 58s/step - loss: 0.6058 - accuracy: 0.4955 - val_loss: 0.5985 - val_accuracy: 0.5063
    Epoch 4/200
    280/280 [==============================] - ETA: 0s - loss: 0.6051 - accuracy: 0.4945 
    '''
    def stain_augmentation(self, image):
        to_augment = staintools.LuminosityStandardizer.standardize(image)
        self.stain_augmentor.fit(to_augment)
        return self.stain_augmentor.pop().astype(np.uint8)



    def decode_example(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.reshape(image_decoded, IMG_SHAPE)
        # https://stackoverflow.com/questions/60347349/attributeerror-tensor-object-has-no-attribute-numpy-in-tensorflow-2-1
        # https://stackoverflow.com/questions/56665868/tensor-numpy-not-working-in-tensorflow-data-dataset-throws-the-error-attribu
        # https://github.com/tensorflow/tensorflow/issues/36979
        if self.stain_variations:  # only supported in tensorflow 2.x
            # Really Really slow TODO: anyway to accelerate?
            image = tf.numpy_function(self.stain_augmentation, [image], [tf.uint8])[0]

        if self.flip_left_right:
            image = tf.image.random_flip_left_right(image)
            # tf.print(image.shape)
        if self.flip_up_down:
            image = tf.image.random_flip_up_down(image)
        if self.color_variations:
            image = tf.image.random_saturation(image, lower=0.2, upper=1.2)
            # image = tf.image.random_brightness(image, max_delta=32./ 255.)
            # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.1)

        image = tf.cast(image, tf.float32) / 255.
        label = tf.cast(tf.one_hot(label, num_classes), tf.int64)
        return image, label

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.file_list, self.label_list))
        dataset = dataset.map(self.decode_example, num_parallel_calls=5)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        return dataset


epochs = 200
patience = 2
bs = 32
trained_models_path = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/model"
log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/log"

VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                        pooling=None, classes=num_classes, classifier_activation='softmax')


# augment_opt = ["stain_variations"]
augment_opt = ["flip_up_down", "flip_left_right", "stain_variations"]
# augment_opt = ["flip_up_down", "flip_left_right", "color_variations"]
# augment_opt = ["flip_up_down", "flip_left_right"]

print(tf.__version__)
for m in all_label_list:
    print("Training " + m + " predictive model")
    # Callbacks

    if not os.path.exists(os.path.join(log_dir, m)):
        os.makedirs(os.path.join(log_dir, m))
    latest_check_point = tf.train.latest_checkpoint(os.path.join(log_dir, m))
    if latest_check_point is not None:
        VGG16_MODEL.load_weights(latest_check_point)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, m), histogram_freq=0, write_graph=True,
                                                  write_images=True)

    model_names = os.path.join(trained_models_path, m, m+'_{epoch:02d}-{val_loss:.4f}.hdf5')
    if not os.path.exists(os.path.join(trained_models_path, m)):
        os.makedirs(os.path.join(trained_models_path, m))
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

    early_stop = EarlyStopping('val_loss', patience=patience)
    # Note: Before Data Augmentation
    # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    # Note:
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=patience, verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Note: Before Data Augmentation
    VGG16_MODEL.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'], run_eagerly=True)
    # Note:
    # VGG16_MODEL.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-6), metrics=['accuracy'], run_eagerly=True)
    # 1e-4: loss: 0.6062 - accuracy: 0.4959 - val_loss: 0.6020 - val_accuracy: 0.5054
    #  loss: 0.6339 - accuracy: 0.3834 - val_loss: 0.6338 - val_accuracy: 0.3848
    #  loss: 0.6070 - accuracy: 0.4333 - val_loss: 0.6086 - val_accuracy: 0.4205
    # 1e-6:
    # loss: 0.5094 - accuracy: 0.5494 - val_loss: 0.5065 - val_accuracy: 0.5487
    # loss: 0.5053 - accuracy: 0.5913 - val_loss: 0.5210 - val_accuracy: 0.5951
    # loss: 0.5478 - accuracy: 0.5670 - val_loss: 0.5485 - val_accuracy: 0.5647

    train_scores_list = pd.Series.get(train_df, m).tolist()
    val_scores_list = pd.Series.get(val_df, m).tolist()

    train_gen = WSI_data_generator(train_file_list, train_scores_list, bs, augment_opt)
    val_gen = WSI_data_generator(val_file_list, val_scores_list, bs, augment_opt)

    # opt = tf.keras.optimizers.Adam(0.1)
    # iterator = iter(train_ds)
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=VGG16_MODEL, iterator=iterator)
    # manager = tf.train.CheckpointManager(ckpt, os.path.join(trained_models_path, m), max_to_keep=3)
    # VGG16_MODEL.save_weights()

    t1 = time.time()
    history = VGG16_MODEL.fit(train_gen.get_dataset(),
                        epochs=epochs,
                        steps_per_epoch=int(training_cnt / bs),
                        validation_steps=int(validate_cnt / bs),
                        validation_data=val_gen.get_dataset(),
                        callbacks=callbacks)

    t2 = time.time() - t1
    K.clear_session()
