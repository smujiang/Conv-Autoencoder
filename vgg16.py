# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential, Model


def vgg16_updated():
    # Encoder
    input_tensor = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((1, 1))(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
    orig_1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 128， 128， 64

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
    orig_2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 64， 64， 128

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)
    orig_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 32， 32， 256

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)
    orig_4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 16， 16， 512

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)
    orig_5 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 8， 8， 512

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv6_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv6_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv6_3')(x)
    orig_6 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 4， 4， 512

    # Add Fully Connected Layer
    x = Flatten(name='flatten')(x)
    x = Dense(8192, activation='relu', name='dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(8192, activation='relu', name='dense2')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='sigmoid', name='softmax')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1, activation='relu', name='dense')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

def vgg16_model(img_rows, img_cols, channel=3):
    model = Sequential()
    # Encoder
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel), name='input'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax', name='softmax'))

    # Loads ImageNet pre-trained data
    weights_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    # weights_path = '/infodev1/non-phi-data/junjiang/Conv-Autoencoder/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    # weights_path = '/infodev1/non-phi-data/junjiang/Conv-Autoencoder/models/model.38-0.0271.hdf5'
    model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    model = vgg16_model(224, 224, 3)
    # input_layer = model.get_layer('input')
    print(model.summary())

    K.clear_session()
