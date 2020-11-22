'''
Patch classification with VGG16 (implemented with tensorflow.keras.application.VGG16)
The patches were extracted from whole slide image of SBOT and HGSOC cases.
In the training and testing phase, image data were fetched directly from tfrecord, which were created before hand by
reading images from folders.

Code was tested on tensorflow-gpu 2.3.1
'''
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import tensorflow.keras.backend as K

label_names = ["SBOT", "HGSOC"]
IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_classes = len(label_names)
epochs = 1
patience = 2

#
# def decode_example(example_proto):
#     features = tf.io.parse_single_example(
#         example_proto,
#         features={
#             'loc_x': tf.io.FixedLenFeature([], tf.int64),
#             'loc_y': tf.io.FixedLenFeature([], tf.int64),
#             'case_id': tf.io.FixedLenFeature([], tf.string),
#             'label': tf.io.FixedLenFeature([], tf.int64),
#             'image_arr': tf.io.FixedLenFeature([], tf.string)
#         }
#     )
#     image = tf.io.decode_raw(features['image_arr'], tf.uint8)
#     # image = tf.cast(image, tf.float32) / 255.
#     image = tf.reshape(image, IMG_SHAPE)
#     label = tf.one_hot(features['label'], num_classes)
#     return image, label

def decode_example(example_proto):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            'loc_x': tf.io.FixedLenFeature([], tf.int64),
            'loc_y': tf.io.FixedLenFeature([], tf.int64),
            'case_id': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_arr': tf.io.FixedLenFeature([], tf.string)
        }
    )
    image = tf.io.decode_raw(features['image_arr'], tf.uint8)
    image = tf.cast(image, tf.float32) / 255. # uncomment this line, if feed into network??
    image = tf.reshape(image, IMG_SHAPE)
    label = tf.one_hot(features['label'], num_classes)
    return image, label


def WSI_data_generator(tf_files, batch_size):
    dataset = tf.data.TFRecordDataset(tf_files)
    dataset = dataset.map(decode_example)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset



training_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/training_tf.csv"
val_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/val_tf.csv"
testing_save_to = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/testing_tf.csv"
train_cnt = len(open(training_save_to, "r").readlines())
val_cnt = len(open(val_save_to, "r").readlines())
test_cnt = len(open(testing_save_to, "r").readlines())

train_tf_file = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256_tfrecords/training_tf_np.tfrecords"
val_tf_file = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256_tfrecords/val_tf_np.tfrecords"
test_tf_file = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256_tfrecords/testing_tf_np.tfrecords"

log_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/log_time/training_from_tfrecord.txt"
fp = open(log_txt, 'a')

'''
Our evaluation suggested that it's better to set batch size larger than 8, otherwise, Tensorflow will throw this warning.
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0432s vs `on_train_batch_end` time: 0.1121s). Check your callbacks.
other conclusion:
1) small batch will take more time on training?
2) small batch will get better trained model(lower training loss)?
batch_size = 4:58039/58039 [==============================] - 2922s 50ms/step - loss: 0.3659 - accuracy: 0.8270 - val_loss: 0.9244 - val_accuracy: 0.6425
batch_size = 8:29019/29019 [==============================] - 2545s 88ms/step - loss: 0.3721 - accuracy: 0.8215 - val_loss: 0.9421 - val_accuracy: 0.6620
batch_size = 16:14509/14509 [==============================] - 2067s 142ms/step - loss: 0.3985 - accuracy: 0.8050 - val_loss: 0.7351 - val_accuracy: 0.6395
batch_size = 32:7254/7254 [==============================] - 1932s 266ms/step - loss: 0.4138 - accuracy: 0.7947 - val_loss: 0.6808 - val_accuracy: 0.7058
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

    train_ds = WSI_data_generator(train_tf_file, batch_size=bs)
    val_ds = WSI_data_generator(val_tf_file, batch_size=bs)
    test_ds = WSI_data_generator(test_tf_file, batch_size=bs)
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
    fp.write("Time elapse for training by getting data (batch_size:%d) from tfrecord %.4f s \n" % (bs, t2))

fp.close()
