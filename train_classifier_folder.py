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

    batch_size_list = [2, 4, 8, 16, 32]
    for batch_size in batch_size_list:
        train_gen = patch_data_label_generator(data_train_txt, batch_size=batch_size)
        val_gen = patch_data_label_generator(data_val_txt, batch_size=batch_size)

        model = vgg16_updated()
        migrate.migrate_model(model)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])
        print(model.summary())

        # Callbacks
        tensor_board = keras.callbacks.TensorBoard(log_dir='./fd_logs_vgg16', histogram_freq=0, write_graph=True,
                                                   write_images=True)
        trained_models_path = 'folder_models/folder_model'
        model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping('val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
        callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

        # Start Fine-tuning
        t1 = time.time()
        model.fit_generator(generator=train_gen,
                            steps_per_epoch=int(train_cnt/batch_size),
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_gen,
                            validation_steps=int(val_cnt/batch_size),
                            verbose=1)
        t2 = time.time() - t1
        K.clear_session()
        print("Time elapse for training by getting data (batch_size:%d) from image folder %.4f s " % (batch_size, t2))

