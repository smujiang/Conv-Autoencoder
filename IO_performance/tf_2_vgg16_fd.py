'''
Patch classification with VGG16 (implemented with tensorflow.keras.application.VGG16)
The patches were extracted from whole slide image of SBOT and HGSOC cases.
In the training and testing phase, image data were fetched directly from image folders.

Code was tested on tensorflow-gpu 2.3.1
'''
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import tensorflow.keras.backend as K
import pandas as pd
import tensorflow as tf
import numpy as np


label_names = ["SBOT", "HGSOC"]
IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = len(label_names)
epochs = 1
patience = 2

training_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/training_tf.csv"
val_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/val_tf.csv"
testing_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/testing_tf.csv"
train_cnt = len(open(training_save_to, "r").readlines())
val_cnt = len(open(testing_save_to, "r").readlines())
test_cnt = len(open(testing_save_to, "r").readlines())


def label_to_int(labels_list, class_index):  # labels is a list of labels
    int_classes = []
    for label in labels_list:
        int_classes.append(class_index.index(label))  # the class_index.index() looks values up in the list label
    int_classes = np.array(int_classes, dtype=np.int32)
    return int_classes, class_index  # returning class index so you know what things are


header = "split,image_url,label\n"
names = header.strip().split(",")

train_df = pd.read_csv(training_save_to, names=names)
val_df = pd.read_csv(val_save_to, names=names)
test_df = pd.read_csv(testing_save_to, names=names)

train_file_list = train_df[names[1]].tolist()[1:]
train_label_txt_list = train_df[names[2]].tolist()[1:]
val_file_list = val_df[names[1]].tolist()[1:]
val_label_txt_list = val_df[names[2]].tolist()[1:]
test_file_list = test_df[names[1]].tolist()[1:]
test_label_txt_list = test_df[names[2]].tolist()[1:]

train_label_list, _ = label_to_int(train_label_txt_list, label_names)
val_label_list, _ = label_to_int(val_label_txt_list, label_names)
test_label_list, _ = label_to_int(test_label_txt_list, label_names)


def decode_example_fd(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32) / 255.
    image = tf.reshape(image, IMG_SHAPE)
    label = tf.cast(tf.one_hot(label, num_classes), tf.int64)
    return image, label


def WSI_data_generator_fd(file_list, label_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((file_list, label_list))
    dataset = dataset.map(decode_example_fd)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


log_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/log_time/training_from_folder.txt"
fp = open(log_txt, 'a')

'''
batch_size = 4:58039/58039 [==============================] - 3472s 60ms/step - loss: 0.3693 - accuracy: 0.8232 - val_loss: 0.9517 - val_accuracy: 0.6227
batch_size = 8:29019/29019 [==============================] - 2921s 101ms/step - loss: 0.3830 - accuracy: 0.8169 - val_loss: 0.7525 - val_accuracy: 0.6728
batch_size = 16:14509/14509 [==============================] - 2395s 165ms/step - loss: 0.3999 - accuracy: 0.8049 - val_loss: 0.7677 - val_accuracy: 0.6580
batch_size = 32:7254/7254 [==============================] - 2347s 324ms/step - loss: 0.4197 - accuracy: 0.7904 - val_loss: 0.6117 - val_accuracy: 0.7236
'''
batch_size_list = [4, 8, 16, 32]
for bs in batch_size_list:
    VGG16_MODEL = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=IMG_SHAPE,
                        pooling=None, classes=num_classes, classifier_activation='softmax')

    # Callbacks
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./tfr_logs_vgg16', histogram_freq=0, write_graph=True,
                                                  write_images=True)
    trained_models_path = 'tf_models/tf_model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    VGG16_MODEL.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])
    # print(VGG16_MODEL.summary())

    train_ds = WSI_data_generator_fd(train_file_list, train_label_list, batch_size=bs)
    val_ds = WSI_data_generator_fd(val_file_list, val_label_list, batch_size=bs)
    test_ds = WSI_data_generator_fd(test_file_list, test_label_list, batch_size=bs)
    t1 = time.time()
    history = VGG16_MODEL.fit(train_ds,
                        epochs=epochs,
                        steps_per_epoch=int(train_cnt / bs),
                        validation_steps=int(val_cnt / bs),
                        # steps_per_epoch=2000,
                        # validation_steps=10,
                        validation_data=val_ds,
                        callbacks=callbacks)

    t2 = time.time() - t1
    K.clear_session()
    fp.write("Time elapse for training by getting data (batch_size:%d) from folder %.4f s \n" % (bs, t2))
fp.close()
