import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.model_selection import train_test_split
from opt import RAdam
import os, csv

#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5,6'

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import efficientnet.keras as efn
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Input, Dense, MaxPooling2D, Conv2D, Flatten, Concatenate, Dropout
from keras.layers.core import Activation, Layer
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras.preprocessing.image import ImageDataGenerator
from icecream import ic

import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
import utils, time

ic("Importing Dataset")
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
using_229 = False
using_my_data = True #'both'
testing = True
class_weight = False
version = '20'   ###450 already has weights
ic(using_229, using_my_data, testing, version, class_weight)
studies = 9000
n_folds = 6
num_epochs = 100
l_test = 5000
l_val = l_test + 5000
prop = l_val /(studies*n_folds + l_val)

if testing:
    num_samples = studies = 100
    n_folds = 2
    num_epochs = 3
    version = 'T'
    l_test = 100
    l_val = l_test + 50

num_samples = studies
study = studies
utils.create_path(study, version)
ic(study)
channels = ['H', 'J', 'Y']

if using_229:
    data_dir = '/home/dados229/cenpes/DataChallenge2/'
    images_vis = np.load(os.path.join(data_dir,'images_efn_vis.npy'))
else:
    data_dir = '/home/dados2T/DataChallenge2/'
    if using_my_data:
        ic(using_my_data)
        images_vis = np.load(os.path.join(data_dir,'images_vis_new.npy'))
    else:
        images_vis = np.load(os.path.join(data_dir, 'images_vis_normalized.npy')) ##images_vis = np.load(os.path.join(data_dir,'images_vis_new_99999.npy'))
ic(data_dir)

with open('code_data_version_%s_aucs_0_%s.csv' % (version, studies), 'w', newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(['train_size', 'min', 'med', 'max'])

with open('code_data_version_%s_aucs_1_%s.csv' % (version, studies), 'w', newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(['train_size', 'min', 'med', 'max'])

is_lens = np.load(os.path.join(data_dir,'Y_2.npy'))
ic(is_lens.shape)
#indexes = utils.shuffler(is_lens)
#is_lens = is_lens[indexes]
is_lens = is_lens[:(l_val+num_samples*n_folds),:]
ic(len(is_lens))

images_vis = utils.img_conv_3(images_vis)
#images_vis = images_vis[indexes]

images_vis = images_vis[:(l_val+num_samples*n_folds),:,:,:]
ic(images_vis.shape)

channels_vis = ['H', 'J', 'Y']

pad = np.zeros((images_vis.shape[0],images_vis.shape[1],images_vis.shape[2],1), dtype="float32")
ic(pad.shape)
X_train_vis2, test_vis2, Y_train2, is_test = train_test_split(images_vis, is_lens, test_size = prop, random_state = 7)
hf = int(0.5*len(is_test))
val_vis = test_vis2[:hf,:,:,:]
test_vis = test_vis2[hf:,:,:,:]
is_val = is_test[:hf]
is_test = is_test[hf:]

ic(X_train_vis2.shape, test_vis2.shape, Y_train2.shape, is_test.shape, is_val.shape)

if using_my_data:
    images_hjy = np.load(os.path.join(data_dir,'images_hjy_new.npy'))
else:
    images_hjy = np.load(os.path.join(data_dir,'images_hjy_normalized.npy'))
ic(images_hjy.shape)
images_hjy = images_hjy[:(l_val+num_samples*n_folds),:,:,:]

#index = utils.print_multi(num_samples, images_hjy, images_vis, num_prints=2, num_channels=images_hjy.shape[3], version=version, input_shape=images_hjy.shape[1], step='BOTH_bf_pp', index=0)
#index = utils.save_clue(images_vis, num_samples, version, 'BOTH_bf_pp', images_vis.shape[1], 3, 3, 0)
utils.histo_all(images_hjy, images_vis, True, channels_vis)

del images_vis

pad = np.zeros((images_hjy.shape[0],images_hjy.shape[1],images_hjy.shape[2],1), dtype="float32")
ic(pad.shape)
X_train_hjy2, test_hjy2, Y_train2, is_test = train_test_split(np.concatenate([images_hjy[:,:,:,2:],pad,pad], axis=-1), is_lens, test_size = prop, random_state = 7)
hf = int(0.5*len(is_test))
val_hjy = test_hjy2[:hf,:,:,:]
test_hjy = test_hjy2[hf:,:,:,:]
is_val = is_test[:hf]
is_test = is_test[hf:]
del images_hjy
ic(X_train_hjy2.shape, test_hjy2.shape, Y_train2.shape, is_test.shape, is_val.shape)

begin = time.perf_counter()

fpr = np.zeros((n_folds, 6), dtype=object)
tpr = np.zeros((n_folds, 6), dtype=object)
roc_auc = np.zeros((n_folds, 6), dtype=object)
ic(fpr.shape, tpr.shape, roc_auc.shape)

for fo in range(n_folds):
    fold = fo
    start = time.perf_counter()

    Y_train = Y_train2[fo*num_samples:((fo+1)*(num_samples)-1)]
    X_train_hjy = X_train_hjy2[fo*num_samples:((fo+1)*(num_samples)-1),:,:,:]
    X_train_vis = X_train_vis2[fo*num_samples:((fo+1)*(num_samples)-1),:,:,:]

    ic("Building Model")

    if class_weight:
        # Scaling by total/2 helps keep the loss to a similar magnitude. 
        # The sum of the weights of all examples stays the same.
        #https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=pt-br
        pos = np.sum(Y_train[:,0])
        neg = Y_train.shape[0] - np.sum(Y_train[:,0])
        ic(pos, neg)
        weight_for_0 = (1 / pos) * (Y_train.shape[0]) / 2.0
        weight_for_1 = (1 / neg) * (Y_train.shape[0]) / 2.0
        Class_WEIGHT = {0: weight_for_0, 1: weight_for_1}

        ic('Weight for class 0: {:.2f}'.format(weight_for_0))
        ic('Weight for class 1: {:.2f}'.format(weight_for_1))


    inp_hjy = Input((66,66,3))
    efn_arc_hjy = efn.EfficientNetB2(input_tensor = inp_hjy, weights='imagenet')

    for layer in efn_arc_hjy.layers:
        efn_arc_hjy.get_layer(layer.name).name = layer.name + "_y"

    inp_vis = Input((200,200,3))
    efn_arc_vis = efn.EfficientNetB2(input_tensor = inp_vis, weights='imagenet')

    for layer in efn_arc_vis.layers:
        efn_arc_vis.get_layer(layer.name).name = layer.name + "_vis"

    concat = Concatenate()([efn_arc_vis.layers[-2].output, efn_arc_hjy.layers[-2].output])

    y_hat = Dense(2,activation="softmax")(concat)  ###originally softmax

    model = Model([efn_arc_vis.input, efn_arc_hjy.input], y_hat)

    plot_model(model, to_file='model.png')

    #multigpu
    model = multi_gpu_model(model, gpus=2)  #ORIGINALLY 5

    model.compile(loss = 'categorical_crossentropy', optimizer=RAdam(),metrics = ['accuracy'])

    ic("Training Model")

    model_name = "efn02_vis_y_01_v_%s_st_%s.hdf5" % (version, study)
    batch_size = 35 *2  ##ORIGINALLY * 5 ##35*2
    if os.path.exists("final_model/" + model_name):
        os.remove("final_model/" + model_name)

    check = ModelCheckpoint("final_model/" + model_name, monitor="val_loss", verbose=1, save_best_only=True)
    ic(X_train_hjy.shape, Y_train.shape)

    ic(np.count_nonzero(Y_train[:, 1] == 1), np.count_nonzero(Y_train[:, 1] == 0), np.count_nonzero(is_val[:, 1] == 1), np.count_nonzero(is_val[:, 1] == 0), np.count_nonzero(is_test[:, 1] == 1), np.count_nonzero(is_test[:, 1] == 0))

    gen = ImageDataGenerator(
		    rotation_range=180,
		    zoom_range=0.20,
		    vertical_flip = True,
            horizontal_flip=True,
		    fill_mode="nearest")

    def gen_flow_for_two_inputs(X1, X2, y):
        genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=1)
        genX2 = gen.flow(X2, batch_size=batch_size,seed=1)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i], X1i[1]

    gen_flow = gen_flow_for_two_inputs(X_train_vis, X_train_hjy, Y_train)

    #model.load_weights("final_model/" + model_name)

    if not class_weight:
        history = model.fit_generator(gen_flow, epochs = num_epochs,  
               verbose = 1, validation_data= ([val_vis, val_hjy], is_val), callbacks=[check], 
               steps_per_epoch = X_train_hjy.shape[0] // batch_size)#, class_weight=Class_WEIGHT)
    else:
        history = model.fit_generator(gen_flow, epochs = num_epochs,  
               verbose = 1, validation_data= ([val_vis, val_hjy], is_val), callbacks=[check], 
               steps_per_epoch = X_train_hjy.shape[0] // batch_size, class_weight=Class_WEIGHT)


    ic("Getting Statistics")

    ic("Training Statistics")
    pred = model.predict([X_train_vis, X_train_hjy])
    
    fig = plt.figure(figsize = (40,20))
    fig.suptitle(model_name[:-5])

    beta = np.sqrt(0.001)
    
    plt.subplot(2,3,1)
    
    #for i in range(2):
        #fpr[fold, i], tpr[fold, i], thresh = roc_curve(Y_train[:, i], pred[:, i])
        #roc_auc[fold, i] = auc(fpr[fold, i], tpr[fold, i])
    fpr[fold, 0], tpr[fold, 0], roc_auc[fold, 0], thresh = utils.roc_curve_calculate(Y_train[:, 0], X_train_vis, pred[:, 0], 'effnetB2')
    fpr[fold, 1], tpr[fold, 1], roc_auc[fold, 1], thresh = utils.roc_curve_calculate(Y_train[:, 1], X_train_hjy, pred[:, 1], 'effnetB2')
    ic(len(fpr[fold, 1]), len(tpr[fold, 1]), roc_auc[fold, 1].shape, len(thresh))
    fs = fpr[fold, 1]
    ts = tpr[fold, 1]
    op = []
    for k in range(len(fs)):
        f2s = fs[k]**2
        t2s = (1-ts[k])**2
        op.append(np.sqrt(f2s + t2s))

    #optimal = np.argmin(np.sqrt((np.array(fpr[fold, 1]))**2 + (np.array(1-tpr[fold, 1]))**2))
    optimal = np.argmin(op)
    optimal_thresh = thresh[optimal]

    FB = fbeta_score(Y_train[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
    FB = FB.max()

    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['~lens','lens']
    for i in range(2):
        plt.plot(fpr[fold, i], tpr[fold, i], color=colors[i],
                 lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[fold, i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    fpr_tresh = fpr[fold, 1]
    tpr_tresh = tpr[fold, 1]
    plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt -{}'.format(optimal_thresh))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Training ROC - Fold {fold} - FBeta {FB}')
    plt.legend(loc="lower right")
    plt.savefig('./graphs/graphs_%s/ver_%s/ROCLensDetectNet_effnet_patrick_0_fold_%s.png' % (study, version, fold))

    ic("Test Statistics")

    pred = model.predict([test_vis, test_hjy])
    ic(pred.shape)
    
    plt.subplot(2,3,2)
    #for i in range(2):
    #fpr[fold, 2+i], tpr[fold, 2+i], thresh = roc_curve(Y_train[:, i], pred[:, i])
    #roc_auc[fold, 2+i] = auc(fpr[fold, 2+i], tpr[fold, 2+i])
        #if i == 0:
        #    fpr[fold, 2+i], tpr[fold, 2+i], roc_auc[fold, 2+i], thresh = utils.roc_curve_calculate(is_test[:, i], test_vis, pred[:, i], 'effnetB2')
        #else:
        #    fpr[fold, 2+i], tpr[fold, 2+i], roc_auc[fold, 2+i], thresh = utils.roc_curve_calculate(is_test[:, i], test_hjy, pred[:, i], 'effnetB2')
        #ic(len(fpr[fold, 2+i]), len(tpr[fold, 2+i]), roc_auc[fold, 2+i].shape, len(thresh))
    fpr[fold, 2], tpr[fold, 2], roc_auc[fold, 2], thresh = utils.roc_curve_calculate(Y_train[:, 0], X_train_vis, pred[:, 0], 'effnetB2')
    fpr[fold, 3], tpr[fold, 3], roc_auc[fold, 3], thresh = utils.roc_curve_calculate(Y_train[:, 1], X_train_hjy, pred[:, 1], 'effnetB2')
    ic(len(fpr[fold, 3]), len(tpr[fold, 3]), roc_auc[fold, 3].shape, len(thresh))
    fs = fpr[fold, 3]
    ts = tpr[fold, 3]
    op = []
    for k in range(len(fs)):
        f2s = fs[k]**2
        t2s = (1-ts[k])**2
        op.append(np.sqrt(f2s + t2s))

    #optimal = np.argmin(np.sqrt((np.array(fpr[fold, 1]))**2 + (np.array(1-tpr[fold, 1]))**2))
    optimal = np.argmin(op)
    optimal_thresh = thresh[optimal]

    FB = fbeta_score(is_test[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
    FB = FB.max() 

    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['~lens','lens']
    for i in range(2):
        plt.plot(fpr[fold, 2+i], tpr[fold, 2+i], color=colors[i],
             lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[fold, 2+i])
    fpr_tresh = fpr[fold, 3]
    tpr_tresh = tpr[fold, 3]
    plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt - {}'.format(optimal_thresh))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Test ROC - Fold {fold} -  FBeta {FB}')
    plt.legend(loc="lower right")
    plt.savefig('./graphs/graphs_%s/ver_%s/ROC2_effnet_patrick_0_fold_%s.png' % (study, version, fold))

    ic("Best Model Statistics")

    #model.load_weights(model_name)
    model.load_weights("final_model/" + model_name)
    pred = model.predict([test_vis, test_hjy])
    
    plt.subplot(2,3,3)
    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #for i in range(2):
        #fpr[fold, 4+i], tpr[fold, 4+i], thresh = roc_curve(is_test[:, i], pred[:, i])
        #roc_auc[fold, 4+i] = auc(fpr[fold, 4+i], tpr[fold, 4+i])
        #if i == 0:
            #fpr[fold, 4+i], tpr[fold, 4+i], roc_auc[fold, 4+i], thresh = utils.roc_curve_calculate(is_test[:, i], test_vis, pred[:, i], 'effnetB2')
        #else:
            #fpr[fold, 4+i], tpr[fold, 4+i], roc_auc[fold, 4+i], thresh = utils.roc_curve_calculate(is_test[:, i], test_hjy, pred[:, i], 'effnetB2')
        #ic(len(fpr[fold, 4+i]), len(tpr[fold, 4+i]), roc_auc[fold, 4+i].shape, len(thresh))
    fpr[fold, 4], tpr[fold, 4], roc_auc[fold, 4], thresh = utils.roc_curve_calculate(Y_train[:, 0], X_train_vis, pred[:, 0], 'effnetB2')
    fpr[fold, 5], tpr[fold, 5], roc_auc[fold, 5], thresh = utils.roc_curve_calculate(Y_train[:, 1], X_train_hjy, pred[:, 1], 'effnetB2')
    ic(len(fpr[fold, 5]), len(tpr[fold, 5]), roc_auc[fold, 5].shape, len(thresh))
    fs = fpr[fold, 5]
    ts = tpr[fold, 5]
    op = []
    for k in range(len(fs)):
        f2s = fs[k]**2
        t2s = (1-ts[k])**2
        op.append(np.sqrt(f2s + t2s))

    #optimal = np.argmin(np.sqrt((np.array(fpr[fold, 1]))**2 + (np.array(1-tpr[fold, 1]))**2))
    optimal = np.argmin(op)
    optimal_thresh = thresh[optimal]

    FB = fbeta_score(is_test[:,1], 1.*(pred[:,1] > optimal_thresh), average=None, beta=beta)
    FB = FB.max()
    
    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['~lens','lens']
    for i in range(2):
        plt.plot(fpr[fold, 4+i], tpr[fold, 4+i], color=colors[i],
             lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[fold, 4+i])
    fpr_tresh = fpr[fold, 5]
    tpr_tresh = tpr[fold, 5]
    plt.plot(fpr_tresh[optimal], tpr_tresh[optimal], '*', label = 'opt - {}'.format(optimal_thresh))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Test ROC (Loading Best Model) - Fold {fold} - FBeta {FB}')
    plt.legend(loc="lower right")
    plt.savefig('./graphs/graphs_%s/ver_%s/ROC_Best_effnet_patrick_0_fold_%s.png' % (study, version, fold))
    

    ic("Else Statistics")

    plt.subplot(2,3,4)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0.0, 1.05])
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(2,3,5)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.ylim([0.0, 1.05])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(f"./graphs/graphs_{study}/ver_{version}/{model_name[:-5]}_fold_{fold}.png")
    #plt.savefig(f"final_model/{model_name[:-5]}.png")
    ic(fpr.shape, tpr.shape, roc_auc.shape)
    #elaps = (time.perf_counter() - fold_start) / 60
    #ic('\n ** Fold TIME: %.3f minutes.' % elaps)
    elapsed = (time.perf_counter() - start) / 60
    ic('\n ** Study TIME: %.3f minutes.' % elapsed)
    del model

thres = int(len(is_test))
threshold_v = np.linspace(1, 0, thres)
titles = ['effnetB2_training', 'effnetB2_testing', 'effnetB2_best']

for j in range(3):
    ic('Usin ultimate %s' % j)
    utils.ultimate_ROC(roc_auc[:,2*j], roc_auc[:,(2*j+1)], threshold_v, tpr[:,2*j], tpr[:,(2*j+1)], fpr[:,2*j], fpr[:,(2*j+1)], titles[j], titles[j], n_folds, study, titles[j], version)

total = (time.perf_counter() - begin) / 60
ic('\n ** Study TIME: %.3f minutes.' % total)

ic(' **** FINISHED!!!')
