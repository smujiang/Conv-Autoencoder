import keras as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
import os
import migrate
# from vgg16_original import vgg16_model_org
from vgg16 import vgg16_updated
from utils import custom_loss
from wsi_data_utils import patch_data_label_generator
import tensorflow as tf
import time
import keras.backend as K
import tensorflow_datasets as tfds
import numpy as np

# tf.enable_eager_execution()

if __name__ == '__main__':
    epochs = 2
    patience = 2


    '''
    # data generator
    '''
    data_train_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256/data_from_folder.txt"
    data_val_txt = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256/data_from_folder.txt"
    train_cnt = len(open(data_train_txt, "r").readlines())
    val_cnt = len(open(data_val_txt, "r").readlines())

    tf_file = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256_tfrecords/testing_tf.tfrecords"

    # def parse_data(orig_dataset):
    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: orig_dataset, {"image_raw": tf.string, "label": tf.int64})
    #     return dataset


    '''
    from keras
    https://gist.github.com/jjallaire/e2efacb54cfc91d554aec2db764632ed
    
    From tensorflow.keras
    https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36   
    
    '''
    img_shape = (256, 256, 3)
    num_classes = 2
    def WSI_dataset(tf_files, batch_size=4):
        def decode_example(example_proto):
            features = tf.parse_single_example(
                example_proto,
                features={
                    'loc_x': tf.io.FixedLenFeature([], tf.int64),
                    'loc_y': tf.io.FixedLenFeature([], tf.int64),
                    'case_id': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                    'image_raw': tf.io.FixedLenFeature([], tf.string)
                }
            )

            image = tf.image.decode_jpeg(features['image_raw'], channels=3)
            image = tf.cast(image, tf.float32) / 255.
            image = tf.reshape(image, img_shape)
            label = tf.one_hot(features['label'], num_classes)

            return image, label

            # return [img_str, label_int]
        dataset = tf.data.TFRecordDataset(tf_files)
        dataset = dataset.map(decode_example)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # next_batch = dataset.make_one_shot_iterator().get_next()
        # while True:
        #     yield K.get_session().run(next_batch)
        return dataset


    # def WSI_data_generator(filenames, batch_size):
    #     dataset = WSI_dataset(filenames, batch_size)
    #     # iter = dataset.make_one_shot_iterator()
    #     # iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    #
    #     # return  iter
    #     batch = iter.get_next()
    #
    #     while True:
    #     #     bo = K.batch_get_value(batch)
    #         yield K.batch_get_value(batch)
    #         # yield batch

    #     image_feature_description = {
    #         'loc_x': tf.io.FixedLenFeature([], tf.int64),
    #         'loc_y': tf.io.FixedLenFeature([], tf.int64),
    #         'case_id': tf.io.FixedLenFeature([], tf.string),
    #         'label': tf.io.FixedLenFeature([], tf.int64),
    #         'image_raw': tf.io.FixedLenFeature([], tf.string)
    #     }
    #
    #     def parse_example(serialized, shape=img_shape):
    #         # Parse the serialized data so we get a dict with our data.
    #         parsed_example = tf.io.parse_single_example(serialized=serialized, features=image_feature_description)
    #         # with tf.Session() as sess:
    #         # label = parsed_example['label']
    #         label = tf.one_hot(tf.cast(image_feature_description['label'], tf.int32), num_classes)
    #         # label = tf.convert_to_tensor([label, abs(label-1)])
    #
    #         # label = tf.one_hot(label, depth=2)
    #
    #         image_raw = parsed_example['image_raw']  # Get the image as raw bytes.
    #         image = tf.decode_raw(image_raw, tf.uint8)  # Decode the raw bytes so it becomes a tensor with type.
    #         image = tf.reshape(image, shape=shape)
    #         return (image, label)
    #
    #
    # d = tf.data.TFRecordDataset([tf_file])
    # d = d.map(parse_example)

    batch_size_list = [4, 8, 16, 32]
    for bs in batch_size_list:
        model = vgg16_updated()
        migrate.migrate_model(model)
        model._make_predict_function()
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])
        model._make_predict_function()
        print(model.summary())

        # Callbacks
        tensor_board = keras.callbacks.TensorBoard(log_dir='./tfr_logs_vgg16', histogram_freq=0, write_graph=True,
                                                   write_images=True)
        trained_models_path = 'tf_models/tf_model'
        model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping('val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
        callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]



        '''
        Data generator 0
        '''
        dataset = WSI_dataset(tf_file, bs)
        # next(dataset)
        # for data in dataset:
        # model.input
        # model.fit_generator(dataset, steps_per_epoch=int(train_cnt / bs), epochs=epochs)

        model.fit_generator(tfds.as_numpy(dataset), steps_per_epoch=int(train_cnt / bs), epochs=epochs)


        '''
        Data generator 1
        '''
        # train_gen = WSI_data_generator(tf_file, bs)
        # val_gen = WSI_data_generator(tf_file, bs)
        #
        # # Start Fine-tuning
        t1 = time.time()
        # # ds = batched_dataset.map(parse_data)
        #
        # # batched_dataset = next(iter(batched_dataset))
        # # model.fit(d, steps_per_epoch=int(train_cnt/batch_size), epochs=epochs)
        # # K.clear_session()
        # with tf.Session() as rsts:
        #     tf.global_variables_initializer().run()
        #     model.fit_generator(generator=train_gen,
        #                     steps_per_epoch=int(train_cnt/bs),
        #                     epochs=epochs,
        #                     callbacks=callbacks,
        #                     # validation_data=val_gen,
        #                     # validation_steps=int(val_cnt/bs),
        #                     verbose=0)


        # model.fit_generator(generator=batched_dataset,
        #                     steps_per_epoch=int(train_cnt/batch_size),
        #                     epochs=epochs,
        #                     callbacks=callbacks,
        #                     validation_data=batched_dataset,
        #                     validation_steps=int(val_cnt/batch_size),
        #                     verbose=0)

        # model.fit(batched_dataset, epochs=epochs)

        K.clear_session()

        t2 = time.time() - t1
        print("Time elapse for training by getting data (batch_size:%d) from tfRecord %.4f s " % (batch_size, t2))

