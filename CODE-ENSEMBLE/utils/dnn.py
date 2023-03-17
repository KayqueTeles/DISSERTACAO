#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:46:12 2019

@author: mbvalentin
"""
""" Basic modules """
import os
import numpy as np
import shutil

""" Keras """
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

""" Architectures """
from .default_architectures import (DummyNet, SequentialNet, FlatNet, XceptionNet,
                                   InceptionNet, ResNet, ResNeXt, UResNet, PatrickNet)

""" Bayesianator """
from .model_conversions import _bayesianize

""" Custom objects """
from .concrete_dropout_wrappers import ConcreteDropout
from .default_architectures import heteroscedastic_loss, challenge_loss

""" To be able to run tensorboard on the background """
from .threading import StoppableThread # for launching tensorboard on background

""" Path handling """
from .io import _check_filepath

""" Json handler """
from .preprocessing import dict_to_catalog
import json

""" Split data into training/validation """
def split_data(inputs, outputs, validation_split = 0.05):
    
    N = [len(inputs[xn]) for xn in inputs][0]
    ids = np.random.permutation(N)
    ntrain = int((1-validation_split)*N)
    XTrain, YTrain = {xn: inputs[xn][ids[:(2*ntrain)]] for xn in inputs}, \
                     {xn: outputs[xn][ids[:(2*ntrain)]] for xn in outputs}
    XVal, YVal = {xn: inputs[xn][ids[(2*ntrain):(ntrain)]] for xn in inputs}, \
                 {xn: outputs[xn][ids[(2*ntrain):(ntrain)]] for xn in outputs}
    Xtes, Ytes = {xn: inputs[xn][ids[ntrain:]] for xn in inputs}, \
                 {xn: outputs[xn][ids[ntrain:]] for xn in outputs}
                 
    return XTrain, YTrain, XVal, YVal, Xtes, Ytes, {'training':ids[:ntrain],
                                        'validation':ids[ntrain:]}


""" Function to build the models """
def build_model(inputs, outputs, 
                architecture = 'ResNet',
                heteroscedastic = False,
                name = 'net',
                cardinality = 32,
                repetitions = [2, 2],
                num_inner_encoders = 3, 
                encoder_units = 64,
                dense_units = 1024,
                loss = 'mse',
                dropout_rate = 0.5,
                weight_regularizer = None,
                dropout_regularizer = None,
                bayesian = True,
                out_units = [512, 256, 128, 64],
               last_activations = None):
    
    """ Get parameters """
    print(inputs)
    input_shape = {xn: inputs[xn].shape for xn in inputs}
    #input_shape = inputs[xn].shape for xn in inputs
    output_shape = {xn: outputs[xn].shape for xn in outputs}
    #output_shape = outputs[xn].shape for xn in outputs
    if last_activations is None:
        last_activations = {xn: 'linear' for xn in outputs}
    
    if (weight_regularizer is None) or (dropout_regularizer is None):
        if input_shape[list(input_shape)[0]][0] is not None:
            nsamples = input_shape[list(input_shape)[0]][0]
            l = 1e-4
            weight_regularizer = l**2. / nsamples
            dropout_regularizer = 2. / nsamples
        else:
            weight_regularizer = 1e-6
            dropout_regularizer = 1e-5
    
    
    """ Get ARCHITECTURE """
    if (architecture.lower() == 'dummy'):
        model = DummyNet(input_shape,  output_shape,
                          last_activations = last_activations, 
                          heteroscedastic = heteroscedastic,
                          name = name,
                          out_units = out_units)
        
    elif (architecture.lower() == 'sequential'):
        model = SequentialNet(input_shape, output_shape,
                              last_activations = last_activations, 
                              heteroscedastic = heteroscedastic,
                              name = name,
                              out_units = out_units)
    
    elif (architecture.lower() == 'flat'):
        model = FlatNet(input_shape, output_shape,
                        last_activations = last_activations,
                        heteroscedastic = heteroscedastic,
                        name = name,
                        out_units = out_units)
        
    elif (architecture.lower() == 'inception'):
        model = InceptionNet(input_shape, output_shape,
                             last_activations = last_activations,
                             heteroscedastic = heteroscedastic,
                             name = name,
                             out_units = out_units)
    
    elif (architecture.lower() == 'xception'):
        model = XceptionNet(input_shape, output_shape,
                            last_activations = last_activations,
                            heteroscedastic = heteroscedastic,
                            name = name,
                            out_units = out_units)
    
    elif (architecture.lower() == 'resnet'):
        model = ResNet(input_shape, output_shape,
                       repetitions = repetitions,
                       last_activations = last_activations,
                       heteroscedastic = heteroscedastic,
                       name = name,
                       out_units = out_units)
    
    elif (architecture.lower() == 'resnext'):
        model = ResNeXt(input_shape, output_shape,
                        cardinality = cardinality,
                        last_activations = last_activations,
                        heteroscedastic = heteroscedastic,
                        name = name,
                        out_units = out_units)
    
    elif (architecture.lower() == 'uresnet'):
        model = UResNet(input_shape, output_shape,
                        num_inner_encoders = num_inner_encoders, 
                        encoder_units = encoder_units,
                        dense_units = dense_units,
                        dropout_rate = dropout_rate,
                        last_activations = last_activations,
                        heteroscedastic = heteroscedastic,
                        name = name,
                        out_units = out_units)
    elif (architecture.lower() == 'patricknet'):
        model = PatrickNet(input_shape,  output_shape,
                          last_activations = last_activations, 
                          heteroscedastic = heteroscedastic,
                           activation = 'elu',
                          name = name,
                          out_units = out_units)

        
    else:
        raise Exception('Undefined model "{}".'.format(architecture))
    
    
    """ Now build the model """
    inputs, outputs, midstream = model.build()
    
    """Get the keras model"""
    model = Model(inputs,outputs)

    """ Obtain bayesian model """
    suffix = '_bayesian'
    lookup_table = {ml.name: ml.name for ml in model.layers}
    if bayesian:
        model, lookup_table, \
        outputs = _bayesianize(model, 
                                weight_regularizer = weight_regularizer, 
                                dropout_regularizer = dropout_regularizer,
                                transfer_weights = False, 
                                verbose = True, suffix = suffix)

    """ Compile model """
    #loss = challenge_loss
    if heteroscedastic:
        loss = heteroscedastic_loss
    model.compile(optimizer = 'nadam', loss = loss, metrics = ['accuracy'])

    """ Set back right name """
    model.name = name

    """ Print model information """
    msg = 'Model "{}" created with the following configuration:\n'.format(model.name)
    # Inputs:
    if isinstance(inputs,list):
        input_shapes_str = ''.join(['\t\t{}: {}\n'.format(ip.name.split(':')[0],
                                      ip.shape[1:]) for ip in inputs])
    else:
        input_shapes_str = '\t\t{}: {}\n'.format(inputs.name.split(':')[0],
                                      inputs.shape[1:])
    msg += '\tInput:\n {}'.format(input_shapes_str)
    # Outputs:
    if isinstance(outputs,list):
        output_shapes_str = ''.join(['\t\t{}: {}\n'.format(ip.name.split(':')[0],
                                      ip.shape[1:]) for ip in outputs])
    else:
        output_shapes_str = '\t\t{}: {}\n'.format(outputs.name.split(':')[0],
                                      outputs.shape[1:])
    msg += '\tOutput:\n {}'.format(output_shapes_str)
    # Architecture
    msg += '\tArchitecture: {}\n'.format(architecture)
    # Number of layers
    nlayers = len(model.layers)
    msg += '\tNumber of layers: {}\n'.format(nlayers)
    # discrimination of layers (counts)
    discriminated_layer_types = np.array([l.__class__.__name__ \
                                          for l in model.layers])
    # discard input layers
    discriminated_layer_types = discriminated_layer_types[\
                                    np.where(discriminated_layer_types != 'InputLayer')[0]]
    discriminated_layer_types, \
    discriminated_layer_counts = np.unique(discriminated_layer_types, 
                                            return_counts = True)
    msg += ''.join(['\t\t<{}> layers: {} ({:3.2f}%)\n'.format(dlt, dlc,100*dlc/nlayers) \
                for (dlt,dlc) in zip(discriminated_layer_types, 
                                    discriminated_layer_counts)])
    
    # Number of parameters
    msg += '\tNumber of network parameters: {}\n'.format(model.count_params())
    # Flags
    msg += '\tBayesian: {}\n'.format(bayesian)
    msg += '\tHeteroscedastic error: {}\n'.format(heteroscedastic)
    print(msg)

    return model, msg

""" function to save model """
def save_model(model, name = 'model', path = os.getcwd()):
    _check_filepath(path)
    
    # serialize model info (loss,etc)
    model_info = {'loss': 'heteroscedastic' if 'heteroscedastic_loss' in str(model.loss)\
                  else model.loss if isinstance(model.loss,str) else 'mse'}
    dict_to_catalog(model_info, os.path.join(path, name), extension = '.info')
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, name + '.json'), 'w') as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, name + '.h5'))
    print("Saved model to disk")

""" function to load model """
def load_model(filename):
    
    # load model info from catalog
    info_file = open(filename+'.info','r')
    info = json.load(info_file)
    
    # load json and create model
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()    
    loaded_model = model_from_json(loaded_model_json, 
                                   custom_objects={'ConcreteDropout':ConcreteDropout})
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    print("Loaded model from disk")
    
    # compile according to loss
    if info['loss'] == 'heteroscedastic':
        loss = heteroscedastic_loss
    else:
        loss = info['loss']
    loaded_model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])

    
    return loaded_model

""" function to predict values from model """
def predict(model, inputs, outputs = None, npost = 300, suffix = ''):
    
    # apply suffix to whole data
    inputs_bay = {xn+suffix: inputs[xn] for xn in inputs}
    # define function to print progress and get estimations from data 
    def get_post(i, bs):
        print('\rProgress: {:3.2f}%'.format(100*i/(npost-1)),
              end='', flush=True)
        return model.predict(inputs_bay, bs)
    
    """ Obtain predictions for whole dataset """
    #N = np.minimum(4000,len(inputs[list(inputs)[0]]))
    N = 100
    
    while N > 1:
        try:
            post_norm = np.array([get_post(i, N) \
                                  for i in range(npost)])
            if post_norm.ndim == 4:
                post_norm = post_norm.transpose((1,0,2,3))
            elif post_norm.ndim == 3:
                post_norm = post_norm[None,:,:,:]
            break
        except:
            N //= 2
            print('Posterior failed... trying with batch size: {}'.format(N))
    
    if outputs is not None:
        post_norm = {xn: post_norm[i] for i,xn in enumerate(outputs)}
        
    return post_norm

""" function to train and evaluate model """
def run_test(model, XTrain, YTrain, inputs, outputs, 
             epochs = 100, batch_size = 32, validation_data = None, 
             npost = 300, suffix = '', early_stopper = True, 
             lr_reducer = True, tensorboard_path = None, data_augmentation = True):
    
    """ Apply suffix to data """
    XTrain = {xn+suffix: XTrain[xn] for xn in XTrain}
    YTrain = {xn+suffix: YTrain[xn] for xn in YTrain}
    
    if validation_data is not None:
        XVal = {xn+suffix: validation_data[0][xn] for xn in validation_data[0]}
        YVal = {xn+suffix: validation_data[1][xn] for xn in validation_data[1]}
        validation_data = (XVal, YVal)
        
    # Get early stopper callback
    cb = []
    if early_stopper:
        cb = [EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=0, 
                            mode='auto')]
    
    if lr_reducer:
        lr = ReduceLROnPlateau(factor=np.sqrt(0.1), 
                               cooldown=0, 
                               patience=5, 
                               min_lr=0.5e-6) # reduces the learning rate according to the training state
        cb.append(lr)
        
    if isinstance(tensorboard_path,str):
        
        #check path exists (if it does, delete old tensorboard folder)
        if os.path.isdir(tensorboard_path):
            shutil.rmtree(tensorboard_path, ignore_errors=True)
        # try to create
        _check_filepath(tensorboard_path)
        
        def launchTensorBoard():
            os.system('tensorboard --logdir=' + tensorboard_path)
            return
        
        """ Start tensorboard """
        t = StoppableThread(target=launchTensorBoard, args=([]))
        t.start()
        
        tb = TensorBoard(log_dir = tensorboard_path,
                         histogram_freq = 0,
                         batch_size = batch_size,
                         write_graph = False, 
                         write_grads = False, 
                         write_images = True)
        cb.append(tb)

    if data_augmentation:
        aug = ImageDataGenerator(
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True)


        # Here is the function that merges our two generators
        # We use the exact same generator with the same random seed for both the y and angle arrays
        def gen_flow_multiinput(XX, YY):
            gens = {xn: aug.flow(XX[xn], YY[list(YY)[0]], batch_size = batch_size, seed = 666) for xn in XX}

            while True:
                xx = {xn: gens[xn].next() for xn in gens}
                yy = {list(YY)[0]: xx[list(xx)[0]][1]}
                yield {xn: xx[xn][0] for xn in xx}, yy

        # Finally create generator
        gen_flow = gen_flow_multiinput(XTrain, YTrain)

        fit_history = model.fit_generator(gen_flow,
                                          epochs = epochs,
            validation_data = validation_data,
            steps_per_epoch = len(XTrain[list(XTrain)[0]]) // batch_size)

    else:

        """ Train model """
        fit_history = model.fit(XTrain, YTrain,
                           epochs = epochs, batch_size = batch_size,
                           validation_data = validation_data,
                           verbose = 1,
                           callbacks = cb)
    
    """ Stop tensorboard (in case it was running) """
    if isinstance(tensorboard_path,str):
        t.stop()
    
    post_norm = predict(model, inputs, outputs = outputs, 
                        npost = npost, suffix = suffix)
    
    # check if we have to return logvars instead of post norms only
    logvars = None
    if np.all([outputs[c].shape[1] != post_norm[c].shape[2] for c in outputs]):
        n = {c: post_norm[c].shape[2]//2 for c in outputs}
        logvars = {c: post_norm[c][:,:,n[c]:] for c in outputs}
        post_norm = {c: post_norm[c][:,:,:n[c]] for c in outputs}
        
    return post_norm, logvars, dict(fit_history.history,**{'epoch':fit_history.epoch})
