from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf
import keras

import efficientnet.keras as efn
from keras.applications.resnet50 import ResNet50
# Todo - Testar a resnet 50 V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from keras.applications import InceptionV3, InceptionResNetV2, Xception
from keras.layers import (Layer, 
                          Input, Dense, Dropout, Lambda, BatchNormalization, 
                          Flatten, Add, Concatenate, 
                          Activation, LeakyReLU,
                          Conv1D, Conv2D, Conv3D,
                          Conv2DTranspose, Conv3DTranspose,
                          SeparableConv1D, SeparableConv2D, 
                          MaxPooling1D, MaxPooling2D, MaxPooling3D, 
                          GlobalAveragePooling1D, GlobalAveragePooling2D, 
                          GlobalAveragePooling3D)

from utils import utils
from icecream import ic


def get_model_effnet(x_data, weights, effnet_version):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    ic("Input Shape Matrix: ", img_shape)
    # ic("X_data Shape: 1- {}, 2- {}, 3- {}".format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    ic('\n ** Utilizando a Rede EfficientNet ', effnet_version)
    ic('\n ** Pesos carregados: ', weights)

    if effnet_version == 'B0':
        effnet = efn.EfficientNetB0(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B1':
        effnet = efn.EfficientNetB1(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B2':
        effnet = efn.EfficientNetB2(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B3':
        effnet = efn.EfficientNetB3(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B4':
        effnet = efn.EfficientNetB4(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B5':
        effnet = efn.EfficientNetB5(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B6':
        effnet = efn.EfficientNetB6(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    else:
        effnet = efn.EfficientNetB7(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    # flat = tf.keras.layers.Flatten()(res_net.output)
    avg_pool = GlobalAveragePooling2D()(effnet.output)
    # dropout = tf.keras.layers.Dropout(0.5)(avg_pool)
    y_hat = Dense(2, activation="sigmoid")(avg_pool)
    model = keras.models.Model(effnet.input, y_hat)

    return model

def compile_model(name_file_rede, model, opt, fold, version):
    ic('\n Compilando rede: ', name_file_rede)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    ic('\n ** Plotting model and callbacks...')
    name_weights = 'Train_model_weights_%s_{epoch:02d}_%s.h5' % (name_file_rede, version)
    csv_name = 'training_{}_fold_{}_ver_{}.csv'.format(name_file_rede, fold, version)
    callbacks = utils.get_callbacks(name_weights=name_weights, patience_lr=10, name_csv=csv_name)

    return callbacks


def fit_model(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks):
    history = model.fit_generator(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks)

    return history

def get_model_resnet(x_data, weights, resnet_depth):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    ic('Input Shape Matrix: ', img_shape)
    # ic('X_data Shape: 1- {}, 2- {}, 3- {}'.format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    ic('\n ** Utilizando a Rede Resnet', resnet_depth)
    ic('\n ** Pesos carregados: ', weights)

    if resnet_depth == 50:

        res_net = ResNet50(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                           pooling=None)
        avg_pool = GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = Dense(2, activation='softmax')(avg_pool)
        model = keras.models.Model(res_net.input, y_hat)

    elif resnet_depth == 101:
        res_net = ResNet101V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
        avg_pool = GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = Dense(2, activation='softmax')(avg_pool)
        model = keras.models.Model(res_net.input, y_hat)

    else:
        res_net = ResNet152V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
        avg_pool = GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = Dense(2, activation='softmax')(avg_pool)
        model = keras.models.Model(res_net.input, y_hat)

    return model

def get_model_inception(x_data, weights, version):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    ic('Input Shape Matrix: ', img_shape)
    # ic('X_data Shape: 1- {}, 2- {}, 3- {}'.format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    ic('\n ** Utilizando a Inception', version)
    ic('\n ** Pesos carregados: ', weights)

    if version == 'V2':
        res_net = InceptionResNetV2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)
    elif version == 'V3':
        res_net = InceptionV3(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)
        
    avg_pool = GlobalAveragePooling2D()(res_net.output)
    y_hat = Dense(2, activation='softmax')(avg_pool)
    model = keras.models.Model(res_net.input, y_hat)

    return model

def get_model_xception(x_data, weights):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    ic('Input Shape Matrix: ', img_shape)
    # ic('X_data Shape: 1- {}, 2- {}, 3- {}'.format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    ic('\n ** Utilizando a Xception')
    ic('\n ** Pesos carregados: ', weights)

    res_net = Xception(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)
        
    avg_pool = GlobalAveragePooling2D()(res_net.output)
    y_hat = Dense(2, activation='softmax')(avg_pool)
    model = keras.models.Model(res_net.input, y_hat)

    return model