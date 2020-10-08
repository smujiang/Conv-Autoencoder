import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
import os
import migrate
from vgg16_original import vgg16_model_org
from utils import custom_loss
from wsi_data_utils import patch_data_label_generator

if __name__ == '__main__':
    batch_size = 16
    epochs = 1000
    patience = 50

    # Load our model
    model = vgg16_model_org(256, 256, 3)
    # migrate.migrate_model(model)
    # model.compile(optimizer='nadam', loss=custom_loss)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=custom_loss)

    print(model.summary())
    # Load our data
    patch_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/auto_enc_patches_256"

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs_vgg16', histogram_freq=0, write_graph=True, write_images=True)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # data generator
    data_train_txt = os.path.join(patch_root, "training_old.txt")
    data_val_txt = os.path.join(patch_root, "validation.txt")
    data_test_txt = os.path.join(patch_root, "testing.txt")
    train_gen = patch_data_label_generator(data_train_txt, batch_size=16)
    val_gen = patch_data_label_generator(data_val_txt, batch_size=16)
    test_gen = patch_data_label_generator(data_test_txt, batch_size=16)

    train_cnt = len(open(data_train_txt, "r").readlines())
    val_cnt = len(open(data_val_txt, "r").readlines())

    # Start Fine-tuning
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=int(train_cnt/batch_size),
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=int(val_cnt/batch_size),
                        verbose=1)


