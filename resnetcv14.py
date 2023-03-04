#pip install numpy keras scikit-learn pydot matplotlib pandas IPython wget tensorflow
#conda install tensorflow-gpu
#pip install wget

import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pylab as py
import scipy.misc
import warnings
import os
import keras
import pydot
import time
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import tarfile
import zipfile
import shutil
import statistics
import random
import h5py

#from astropy.io import fits

#####3
from tensorflow.keras import applications
from tensorflow.keras.applications import vgg16
#####3

from tensorflow.keras import models

####
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
####

from tensorflow.keras.layers import Input

#########
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
#######

from pathlib import Path
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import KFold  #para cross-validation
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn import metrics

########################################################
from utilities03 import fileremover, filemover, load_data_kfold, get_model, get_callbacks, HighestInteger, ROCCurveCalculate, data_downloader, TestSamplesBalancer, FScoreCalc, remove_weights, last_mover
########################################################

warnings.filterwarnings("ignore")
#print("Num GPUs Available: ", len(tf.config.experimental.set_visible_devices('gpu')))
print("Num GPUs Available: ", len(tf.test.gpu_device_name()))

print("\n ## Tensorflow version:")
print(tf.__version__)

print(" ## Is GPU available?")
print(tf.test.is_gpu_available())

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

###############################################PARAMETERS

num_epochs = 50
batch_size = 64
nk = 10 #10    #NÚMERO DE FOLDS
version = 1        #VERSÃO PRA COLOCAR NAS PASTAS
PC = 1.0     #PORCENTAGEM DO DATASET DE TESTE A SER USADO
num_classes = 2
classes = ['lens', 'not-lens']
width = 0.35
vallim = 2000
PARAM = 0

########################################################

#tf.debugging.set_log_device_placement(True)
##########################################33

print(" ** Cleaning up previous files...")
if os.path.exists('GRAPHS'):
    shutil.rmtree('GRAPHS')

print(" ** Verifying data...")

data_downloader()
print(" ** Reading data from y_data20000fits.csv...")

#LOAD Y_DATA
PATH = os.getcwd()
var = PATH + "/" + "lensdata/"
y_batch = os.listdir(var)

#HERE WE'RE GOING TO START MULTIPLE PROGRAMS!
study = [900, 800, 700, 600, 500, 450, 400, 350, 300, 275, 250, 225, 200, 175, 1500, 1400, 1300, 1250, 1200, 1150, 1100, 1050, 1000, 950, 900, 850, 800, 750, 700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 490, 480, 470, 460, 450, 440, 430, 420, 410, 400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
print(study)

begin = time.perf_counter()
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
Hlauc, Hhauc, Hmauc, GlobF1s, GlobF001s, GloblF1s, GloblF001s, GlobhF1s, GlobhF001s = ([] for i in range(9))

for u in range(0,len(study)):
    try:
        TR = study[u]
        if TR == 175:
            PARAM = 1

        print('\n\n\n ** NEW CICLE WITH %s TRAINING SAMPLES! **************************************************************************************************' % TR)
        ####################################3

        print('\n ** Cleaning up previous files and folders...')
        fileremover(TR, nk, version)       
        ######################################################
       
        print("\n ** Starting data preprocessing...")

        labels = pd.read_csv(var + 'y_data20000fits.csv',delimiter=',', header=None)
        y_data = np.array(labels, np.uint8)
        y_size = len(y_data)
        y_data.shape

        x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')
        Ni_channels = 0 #first channel
        N_channels = 3 #number of channels

        x_data = x_datasaved['data']
        x_size = len(x_data)
        x_data = x_data[:,:,:,Ni_channels:Ni_channels + N_channels]

        print(" ** Randomizing y_data and x_data...")
        ind = np.arange(y_data.shape[0])
        np.random.shuffle(ind)
        y_data = y_data[ind]
        x_data = x_data[ind]

        print(" ** y_data has shape: ", y_data.shape)
        print(" ** Total dataset size: ", y_size, "objects.")

        print(' ** Balancing number of samples on each class for train+val sets with %s samples...' % TR)     

        y_data, x_data, y_test, x_test, y_val, x_val = TestSamplesBalancer(y_data, x_data, vallim, TR, nk)

        y_size = len(y_data)
        y_tsize = len(y_test)
        x_size = len(x_data)

        #PC = AMOUNT OF THE DATASET USED (< 1 FOR TESTING)
        #y_val = y_val[0:int(y_size*PC),]
        #x_val = x_val[0:int(y_size*PC),:,:,:]
        #y_test = y_test[0:int(y_size*PC),]
        #x_test = x_test[0:int(y_size*PC),:,:,:]
        ##############    

        print(" ** y_data arranged with format:")
        print(" ** y_test:   ", y_test.shape)
        print(" ** y_data:  ", y_data.shape)
        print(" ** y_val:  ", y_val.shape)

        #############DISTRIBUTION GRAPH#########
        trainval_count = [np.count_nonzero(y_data == 1)+np.count_nonzero(y_val == 1), np.count_nonzero(y_data == 0)+np.count_nonzero(y_val == 0)]
        test_count = [np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)]

    #############DISTRIBUTION GRAPH#########
        plt.figure()
        fig, ax = plt.subplots()
        ax.bar(classes, test_count, width, label='Test')
        ax.bar(classes, trainval_count, width, bottom=test_count, label='Train+Val')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset distribution')
        ax.legend(loc='lower right')
        fig.savefig("TrainTest_rate_TR_{}.png". format(TR))
        ##############################333

        print("\n ** x_data splitted with format:")
        print(" ** x_test:   ", x_test.shape)
        print(" ** x_data:  ", x_data.shape)
        print(" ** x_val:  ", x_val.shape)

        print("\n ** Converting data and list of indices into folds for cross-validation...")

        subset_size = int(y_size/nk)   
        folds = load_data_kfold(nk, x_data, y_data)

        print("\n ** Starting network training... \n")

        start = time.perf_counter()
        FPRall, TPRall, AUCall, acc0, loss0, val_acc0, val_loss0, lauc, f1s, f001s = ([] for i in range(10))
        y_test = to_categorical(y_test,num_classes=2)

        for j, (train_idx, val_idx) in enumerate(folds):
    
            remove_weights(TR, nk, version)
            foldtimer = time.perf_counter()
            print('\n ** Fold: %s with %s training samples' % (j, TR))
            x_val_cv = x_val
            y_val_cv = y_val
            if PARAM == 1:
                x_data_cv = x_data[val_idx]
                y_data_cv= y_data[val_idx]
            else:
                x_data_cv = x_data[train_idx]
                y_data_cv = y_data[train_idx]

            #############DISTRIBUTION GRAPH#########
            train_count = [np.count_nonzero(y_data_cv == 1), np.count_nonzero(y_data_cv == 0)]
            val_count = [np.count_nonzero(y_val_cv == 1), np.count_nonzero(y_val_cv == 0)]

            #############DISTRIBUTION GRAPH#########
            plt.figure()
            fig, ax = plt.subplots()
            ax.bar(classes, train_count, width, label='Train')
            ax.bar(classes, val_count, width, bottom=train_count, label='Validation')
            ax.set_ylabel('Number of Samples')
            ax.set_title('Data distribution on fold %s with %s training samples)' % (j, TR))
            ax.legend(loc='lower right')
            fig.savefig("TrainVal_rate_TR_{}_Fold_{}.png". format(TR, j))     
            ########################################

            print("\n ** Converting vector classes to binary matrices...")
            y_data_cv = to_categorical(y_data_cv,num_classes=2)
            y_val_cv = to_categorical(y_val_cv,num_classes=2)

            print("\n ** Building ResNet model...")
    
            model = get_model(x_data, y_data, N_channels)
            model.summary()
            print("\n ** Compiling model...")
            lr = 0.01
            sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            #from keras.utils import multi_gpu_model
            #model = multi_gpu_model(model, gpus=2)
            #with tf.device("/GPU:0"):
            model.compile(loss= 'binary_crossentropy' , optimizer='sgd' , metrics=[ 'accuracy' ])

            print("\n ** Plotting model and callbacks...")
                #plot_model(model,  to_file="model_ResNet50.png")
            gen = ImageDataGenerator(rotation_range = 90)

            name_weights = "Train_model_weights_{epoch:02d}.h5"
            csv_name = "training_k.csv"
            callbacks = get_callbacks(name_weights = name_weights, patience_lr=10, name_csv = csv_name)
            generator = gen.flow(x_data_cv, y_data_cv, batch_size = batch_size)
            history = model.fit_generator(
                          generator,
                          steps_per_epoch=len(x_data_cv)/batch_size,
                          epochs=num_epochs,
                          verbose=1,
                          validation_data = (x_val_cv, y_val_cv),
                          validation_steps = len(x_val_cv)/batch_size,
                          callbacks = callbacks)
      
      
            print("\n ** Training completed.")            
    
            accu = history.history['accuracy']
            accu_val = history.history['val_accuracy']
            c = HighestInteger(accu, num_epochs)

            print("\n ** Plotting training & validation accuracy values.")  
            plt.figure()
            plt.xlim([0,num_epochs])
            plt.ylim([0,c])
            plt.plot(accu)
            plt.plot(accu_val)
            plt.title('Model accuracy' )
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.savefig("AccxEpoch_{}_Fold_{}.png". format(TR, j))
 
            loss = history.history['loss']
            loss_val = history.history['val_loss']
            c = HighestInteger(loss, num_epochs)

            print("\n ** Plotting training & validation loss values.")
            plt.figure()
            plt.xlim([0,num_epochs])
            plt.ylim([0,c])
            plt.plot(loss)
            plt.plot(loss_val)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.savefig("LossxEpoch_{}_Fold_{}.png". format(TR, j))
            print("\n ** Model evaluation stage.")

            tpr, fpr, auc, auc2, thres = ROCCurveCalculate(y_test, x_test, model)
            lauc = np.append(lauc, auc)
            AUCall.append(auc2)
            FPRall.append(fpr)
            TPRall.append(tpr)

            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--') # k = color black
            plt.plot(FPRall[j], TPRall[j], label="fold" + str(j) + "& AUC: %.3f" % auc, color='C'+str(j), linewidth=3) # for color 'C'+str(j), for j[0 9]
            plt.legend(loc='lower right', ncol=1, mode="expand")
            plt.title('ROC for %s training samples on fold %s' % (TR, j))
            plt.xlabel('false positive rate', fontsize=14)
            plt.ylabel('true positive rate', fontsize=14)
    
            plt.savefig("ROCLensDetectNet_{}_Fold_{}.png". format(TR, j))

            f1_score, f001_score = FScoreCalc(y_test, x_test, model)
            f1s = np.append(f1s, f1_score)
            f001s = np.append(f001s, f001_score)

            acc0 = np.append(acc0,history.history['accuracy'])
            val_acc0 = np.append(val_acc0,history.history['val_accuracy'])
            loss0 = np.append(loss0,history.history['loss'])
            val_loss0 = np.append(val_loss0,history.history['val_loss'])

            scores = model.evaluate(x_test, y_test, verbose=0)
            print(" ** Large CNN Error: %.2f%%" % (100-scores[1]*100))
            elaps = (time.perf_counter() - foldtimer)/60
            print('\n ** Fold TIME: %.3f minutes.' % elaps)

            K.clear_session() 

        print('\n ** Training and evaluation complete.')
        elapsed = (time.perf_counter() - start)/60
        print(' ** %.3f TIME: %.3f minutes.' % (TR,elapsed))

        print('\n ** Generating ultimate ROC graph...')
        medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--') # k = color black

        mauc = np.percentile(lauc, 50.0)
        mAUCall = np.percentile(AUCall, 50.0)
        plt.title('Median ROC over %s folds with %s training samples' % (nk, TR))
        plt.xlabel('false positive rate', fontsize=14)
        plt.ylabel('true positive rate', fontsize=14)

        for num in range(0,int(thres),1):
            lis = [item[num] for item in TPRall]
            los = [item[num] for item in FPRall]
            
            medians_x.append(np.percentile(los, 50.0))
            medians_y.append(np.percentile(lis, 50.0))
            lowlim.append(np.percentile(lis, 15.87))
            highlim.append(np.percentile(lis, 84.13))
        
        lowauc = metrics.auc(medians_x, lowlim)
        highauc = metrics.auc(medians_x, highlim)
        print('\n\n\n ** IS THIS CORRECT?')
        print(lowauc, mauc, highauc)
        print(lowauc, mAUCall, highauc)

        plt.plot(medians_x, medians_y, 'b', label = 'AUC: %s' % mauc, linewidth=3)  
        plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
        plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
        plt.legend(loc='lower right', ncol=1, mode="expand")

        plt.savefig("ROCLensDetectNet_Full_%s.png" % TR)

        print(' ** Generating F1s and F001 scores graph...')

        xey = np.linspace(0, nk, 10)

        plt.figure()
        plt.ylim([0,c])
        plt.plot(xey, f1s, linewidth=3)
        plt.title('F1 Scores per Fold on %s training samples' % TR)
        plt.ylabel('F1 Scores')
        plt.xlabel('Fold')
        plt.savefig("F1xFold_%s.png" % TR)

#######################################################

        plt.figure()
        plt.ylim([0,c])
        plt.plot(xey, f001s, linewidth=3)
        plt.title('F0.01 Scores per Fold on %s training samples' % TR)
        plt.ylabel('F0.01 Score')
        plt.xlabel('Fold')
        plt.savefig("F001xFold_%s.png" % TR)

        mf1score = np.percentile(f1s, 50.0)
        mf001score = np.percentile(f001s, 50.0)
        lf1score = np.percentile(f1s, 15.87)
        lf001score = np.percentile(f001s, 15.87)
        hf1score = np.percentile(f1s, 84.13)
        hf001score = np.percentile(f001s, 84.13)

        filemover(TR, version, nk) 

        Hlauc = np.append(Hlauc, lowauc)
        Hhauc = np.append(Hhauc, highauc)
        Hmauc = np.append(Hmauc, mauc)
        GlobF1s = np.append(GlobF1s, mf1score)
        GlobF001s = np.append(GlobF001s, mf001score)
        GloblF1s = np.append(GloblF1s, lf1score)
        GloblF001s = np.append(GloblF001s, lf001score)
        GlobhF1s = np.append(GlobhF1s, hf1score)
        GlobhF001s = np.append(GlobhF001s, hf001score)

    except AssertionError as error:
        print(error)
    #except:
     #   pass

print('\n ** Executing final evaluations...')

#fileremover(TR, nk, version)

print(' ** Creating data_frames ...')
arr = np.array(study, Hmauc, Hlauc, Hhauc)
pd.DataFrame(arr).to_csv("AUC_data.csv")
print(arr)
arrb = np.array(study, f1s, f001s)
pd.DataFrame(arrb).to_csv("f1_scores.csv")
print(arrb)

print(' ** Building and plotting final graphs...')
study.append(Hlauc)
study.append(Hhauc)
study.append(Hmauc)
c = 1.0

err = [np.absolute(np.subtract(study[3][:], study[1][:])), np.absolute(np.subtract(study[3][:], study[2][:]))]

plt.figure()
plt.ylim([0,c])
# TODO - N = len(study)
#K = np.linspace(0.5, 0.5, N)
x2 = np.arange(study[0][:])
#plt.plot(X3, Y3, color='B', linewidth=3)
#plt.plot(X, g, '-')
plt.plot(x2, study[3][:], 'b', linewidth=3)
plt.errorbar(x2, study[3][:], yerr=err, fmt='o', capsize = 2, color='B', linewidth=1)
plt.xticks(x2, study[0][:])
plt.xticks(rotation=90)
ax = plt.gca()
for label in ax.get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
#plt.xticks(np.arange(0,20, 1))
plt.title('AUC x Training fold set size' )
plt.ylabel('Area under curve (AUC)')
plt.xlabel('Training set size')
plt.savefig("AUCxSize.png")

plt.figure()
plt.plot([0, 1], [0, 1], 'k--') # k = color black

plt.title('Median F1 Scores for %s samples' % (TR))
plt.xlabel('Training set size', fontsize=14)
plt.ylabel('F1 scores', fontsize=14)
plt.plot(x2, GlobF1s, 'b', linewidth=3)
plt.fill_between(x2, GlobF1s, GloblF1s, color='blue', alpha=0.3, interpolate=True)
plt.fill_between(x2, GlobF1s, GlobhF1s, color='blue', alpha=0.3, interpolate=True)

plt.savefig("F1Scores_Full.png")

plt.figure()
plt.plot([0, 1], [0, 1], 'k--') # k = color black

plt.title('Median F001 Scores for %s samples' % (TR))
plt.xlabel('Training set size', fontsize=14)
plt.ylabel('F001 scores', fontsize=14)
plt.plot(x2, GlobF001s, color='B', linewidth=3)
plt.fill_between(x2, GlobF001s, GloblF001s, color='blue', alpha=0.3, interpolate=True)
plt.fill_between(x2, GlobF001s, GlobhF001s, color='blue', alpha=0.3, interpolate=True)

plt.savefig("F001Scores_Full.png")

lastmover(TR, version, nk) 

timee = (time.perf_counter() - begin)/(60*60)
print('\n ** Mission accomplished in %s hours.' % timee)
print("\n ** FINISHED! ************************")
