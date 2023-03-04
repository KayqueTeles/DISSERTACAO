""" Utility functions. """
import numpy as np
import os
import random
import shutil
import sklearn
import keras
import wget
import zipfile
import tarfile
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from collections import Counter

def remove_weights(TR, nk, version):

    weicounter, csvcounter = (0 for i in range(2))
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .h5 and csv files...')
    for bu in range(0, 10*nk, 1):    
        if os.path.exists('Train_model_weights_%s.h5' % bu):
            os.remove('Train_model_weights_%s.h5' % bu)
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights_0{}.h5'. format(bu)):
            os.remove('Train_model_weights_0{}.h5'. format(bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights.h5'):
            os.remove('Train_model_weights.h5')
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights_{}.h5'. format(TR, bu)):
            os.remove('Train_model_{}_weights_{}.h5'. format(TR, bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights_0{}.h5'. format(TR, bu)):
            os.remove('Train_model_{}_weights_0{}.h5'. format(TR, bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights.h5'. format(TR)):
            os.remove('Train_model_{}_weights.h5'. format(TR))
            weicounter = weicounter + 1
        if os.path.exists('trainning_k.csv'):
            os.remove('trainning_k.csv')
            csvcounter = csvcounter + 1
    print(" ** Done. %s .h5 files removed and %s .csv removed." % (weicounter, csvcounter))

def fileremover(TR, nk, version):

    piccounter, weicounter, csvcounter = (0 for i in range(3))
    print('\n ** Removing specified files and folders...')
    for bu in range(0, 10*nk, 1):
        if os.path.exists('./AccxEpoch_{}_Fold_{}.png'. format(TR, bu)):
            os.remove('./AccxEpoch_{}_Fold_{}.png'. format(TR, bu))
            piccounter = piccounter + 1
        if os.path.exists('./TrainTest_rate_TR_{}.png'. format(TR)):
            os.remove('./TrainTest_rate_TR_{}.png'. format(TR))
            piccounter = piccounter + 1
        if os.path.exists('./TrainVal_rate_TR_{}_Fold_{}.png'. format(TR, bu)):
            os.remove('./TrainVal_rate_TR_{}_Fold_{}.png'. format(TR, bu))
            piccounter = piccounter + 1
        if os.path.exists('./LossxEpoch_{}_Fold_{}.png'. format(TR, bu)):
            os.remove('./LossxEpoch_{}_Fold_{}.png'. format(TR, bu))
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_{}_Fold_{}.png'. format(TR, bu)):
            os.remove('./ROCLensDetectNet_{}_Fold_{}.png'. format(TR, bu))
            piccounter = piccounter + 1
        if os.path.exists('./F1xFold_%s.png' % TR):
            os.remove('./F1xFold_%s.png' % TR)
            piccounter = piccounter + 1
        if os.path.exists('./F001xFold_%s.png' % TR):
            os.remove('./F001xFold_%s.png' % TR)
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_Full_%s.png' % TR):
            os.remove('./ROCLensDetectNet_Full_%s.png' % TR)
            piccounter = piccounter + 1
        if os.path.exists('./AUCxSize.png'):
            os.remove('./AUCxSize.png')
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_All_%s.png' % TR):
            os.remove('./ROCLensDetectNet_All_%s.png' % TR)
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_All_Full_%s.png' % TR):
            os.remove('./ROCLensDetectNet_All_Full_%s.png' % TR)
            piccounter = piccounter + 1
        if os.path.exists('Train_model_weights_%s.h5' % bu):
            os.remove('Train_model_weights_%s.h5' % bu)
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights_0{}.h5'. format(bu)):
            os.remove('Train_model_weights_0{}.h5'. format(bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights.h5'):
            os.remove('Train_model_weights.h5')
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights_{}.h5'. format(TR, bu)):
            os.remove('Train_model_{}_weights_{}.h5'. format(TR, bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights_0{}.h5'. format(TR, bu)):
            os.remove('Train_model_{}_weights_0{}.h5'. format(TR, bu))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_{}_weights.h5'. format(TR)):
            os.remove('Train_model_{}_weights.h5'. format(TR))
            weicounter = weicounter + 1
        if os.path.exists('trainning_k.csv'):
            os.remove('trainning_k.csv')
            csvcounter = csvcounter + 1
        if os.path.exists('GRAPHS/RNCV_%s_%s' % (version, TR)):
            shutil.rmtree('GRAPHS/RNCV_%s_%s' % (version, TR))

    print(" ** Removing done. %s .png files removed, %s .h5 files removed and %s .csv removed." % (piccounter, weicounter, csvcounter))

    if os.path.exists('GRAPHS/RNCV_%s_%s' % (version, TR)):
        shutil.rmtree('GRAPHS/RNCV_%s_%s' % (version, TR))

def last_mover(TR, version, nk):
    print(' ** Initiating last_mover...')
    if os.path.exists('./AUCxSize.png'):
        shutil.move('./AUCxSize.png', dest1)
        counter = counter + 1
    if os.path.exists('./AUC_data.csv'):
        shutil.move('./AUC_data.csv', dest1)
        counter = counter + 1
    if os.path.exists('f1_scores.csv'):
        os.remove('f1_scores.csv')
        counter = counter + 1
    if os.path.exists('./F1Scores_Full.png'):
        shutil.move('./F1Scores_Full.png', dest1)
        counter = counter + 1
    if os.path.exists('./F001Scores_Full.png'):
       shutil.move('./F001Scores_Full.png', dest1)
       counter = counter + 1

    print(" ** Moving done. %s files moved." % counter)
    

def filemover(TR, version, nk):

    print('\n ** Moving created files to a certain folder.')
    counter = 0
    print(" ** Checking if there's a GRAPHS folder...")
    if os.path.exists('GRAPHS'):
        print(" ** GRAPHS file found. Moving forward.")
    else:
        print(" ** None found. Creating one.")
        os.mkdir('GRAPHS')
        print(" ** Done!")
    print(" ** Checking if there's an RNCV folder...")
    if os.path.exists('GRAPHS/RNCV_%s_%s' % (version, TR)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree('GRAPHS/RNCV_%s_%s' % (version, TR))
        os.mkdir('GRAPHS/RNCV_%s_%s' % (version, TR))
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir('GRAPHS/RNCV_%s_%s' % (version, TR))
        print(" ** Done!")

        dest1 = ('/home/kayque/LENSLOAD/GRAPHS/RNCV_%s_%s/' % (version, TR))

    for bu in range(0, 10*nk, 1):
        if os.path.exists('./AccxEpoch_{}_Fold_{}.png'. format(TR, bu)):
            shutil.move('./AccxEpoch_{}_Fold_{}.png'. format(TR, bu), dest1)
            counter = counter + 1
        if os.path.exists('./LossxEpoch_{}_Fold_{}.png'. format(TR, bu)):
            shutil.move('./LossxEpoch_{}_Fold_{}.png'. format(TR, bu), dest1)
            counter = counter + 1
        if os.path.exists('./TrainTest_rate_TR_{}.png'. format(TR)):
            shutil.move('./TrainTest_rate_TR_{}.png'. format(TR), dest1)
            counter = counter + 1
        if os.path.exists('./TrainVal_rate_TR_{}_Fold_{}.png'. format(TR, bu)):
            shutil.move('./TrainVal_rate_TR_{}_Fold_{}.png'. format(TR, bu), dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_{}_Fold_{}.png'. format(TR, bu)):
            shutil.move('./ROCLensDetectNet_{}_Fold_{}.png'. format(TR, bu), dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_Test_%s.png' % TR):
            shutil.move('./ROCLensDetectNet_Test_%s.png' % TR, dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_Full_%s.png' % TR):
            shutil.move('./ROCLensDetectNet_Full_%s.png' % TR, dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_All_%s.png' % TR):
            shutil.move('./ROCLensDetectNet_All_%s.png' % TR, dest1)
            counter = counter + 1
        if os.path.exists('./AUCxSize.png'):
            shutil.move('./AUCxSize.png', dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_All_Full_%s.png' % TR):
            shutil.move('./ROCLensDetectNet_All_Full_%s.png' % TR, dest1)
            counter = counter + 1
        if os.path.exists('trainning_k.csv'):
            os.remove('trainning_k.csv')
            counter = counter + 1
        if os.path.exists('./F1xFold_%s.png' % TR):
            shutil.move('./F1xFold_%s.png' % TR, dest1)
            counter = counter + 1
        if os.path.exists('./F001xFold_%s.png' % TR):
            shutil.move('./F001xFold_%s.png' % TR, dest1)
            counter = counter + 1

    print(" ** Moving done. %s files moved." % counter)

def load_data_kfold(k, x_data, y_data):

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_data, y_data))
    
    return folds

def get_model(x_data, y_data, N_channels):

    K.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    img_input = Input(shape=img_shape)
    if N_channels == 3:
        model = ResNet50(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
    else:
        model = ResNet50(include_top=False, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)

    return model

def get_callbacks(name_weights, patience_lr, name_csv):

    mcp_save = ModelCheckpoint(name_weights)
    csv_logger = CSVLogger(name_csv)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='max')
    return [mcp_save, csv_logger, reduce_lr_loss]

def HighestInteger(lst, num_epochs):

    c = 1
    for i in range(0,num_epochs,1):
        if lst[i] > c:
            c = int(round(lst[i]))
    return c

def ROCCurveCalculate(y_test, x_test, model):

    probs = model.predict(x_test)
    probsp = probs[:, 1]
    y_new = y_test[:, 1]
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))
    
    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        TPscore, FPscore, TNscore, FNscore = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:                
                    TPscore = TPscore + 1
                else:
                    FPscore = FPscore + 1
            else:
                if y_new[xz] == 0:
                    TNscore = TNscore + 1
                else:
                    FNscore = FNscore + 1
        TPRate = TPscore / (TPscore + FNscore)
        FPRate = FPscore / (FPscore + TNscore)
        tpr.append(TPRate)
        fpr.append(FPRate)           

    auc2 = roc_auc_score(y_test[:,1], probsp)
    auc = metrics.auc(fpr, tpr)
    print('\n ** AUC (via metrics.auc): %s, AUC (via roc_auc_score): %s' % (auc, auc2))
    return [tpr, fpr, auc, auc2, thres]

def data_downloader():
    print('\n ** Checking files...')
    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(" ** Files from lensdata.tar.gz were already downloaded.")
    else:
        print("n ** Downloading lensdata.zip...")
        wget.download('https://clearskiesrbest.files.wordpress.com/2019/02/lensdata.zip')
        print(" ** Download successful. Extracting...")
        with zipfile.ZipFile("lensdata.zip", 'r') as zip_ref:
            zip_ref.extractall() 
            print(" ** Extracted successfully.")
        print(" ** Extracting data from lensdata.tar.gz...")
        tar = tarfile.open("lensdata.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print(" ** Extracted successfully.")
    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(" ** Files from lensdata.tar.gz were already extracted.")
    else:
        print(" ** Extracting data from #DataVisualization.tar.gz...")     
        tar = tarfile.open("./lensdata/DataVisualization.tar.gz", "r:gz")
        tar.extractall("./lensdata/")
        tar.close()
        print(" ** Extracted successfully.")
        print(" ** Extrating data from x_data20000fits.h5.tar.gz...")     
        tar = tarfile.open("./lensdata/x_data20000fits.h5.tar.gz", "r:gz")
        tar.extractall("./lensdata/")
        tar.close()
        print(" ** Extracted successfully.") 
    if os.path.exists('lensdata.tar.gz'):
            os.remove('lensdata.tar.gz')
    if os.path.exists('lensdata.zip'):
            os.remove('lensdata.zip')
    for pa in range(0, 10, 1):
        if os.path.exists('lensdata ({}).zip'. format(pa)):
            os.remove('lensdata ({}).zip'. format(pa))

def TestSamplesBalancer(y_data, x_data, vallim, TR, nk):
   
    print(" ** Initiating RegularBalancer")
    y_size = len(y_data)
    y_yes, y_no, y_excess = ([] for i in range(3))
    for y in range(0,y_size,1):
        if y_data[y] == 1:
            if len(y_yes)<(TR*(nk/2)):
                y_yes = np.append(int(y), y_yes)
            else: 
                y_excess = np.append(int(y), y_excess)
        else:
            if len(y_no)<(TR*(nk/2)):
                y_no = np.append(int(y), y_no)
            else: 
                y_excess = np.append(int(y), y_excess)
    
    y_y = np.append(y_no, y_yes)
    np.random.shuffle(y_y)

    np.random.shuffle(y_excess)
    y_y = y_y.astype(int)
    y_excess = y_excess.astype(int)

    y_val = y_data[y_excess[0:vallim]]
    x_val = x_data[y_excess[0:vallim]]

    y_test = y_data[y_excess[vallim:int(len(y_excess))]]
    x_test = x_data[y_excess[vallim:int(len(y_excess))]]
    
    y_data = y_data[y_y]
    x_data = x_data[y_y]

    return [y_data, x_data, y_test, x_test, y_val, x_val]

def OverBalancer(y_data, x_data, vallim, TR, nk):
   
    print(" ** Initiating OVERBalancer")
    y_size = len(y_data)
    y_yes, y_no, y_excess = ([] for i in range(3))
    for y in range(0,y_size,1):
        if len(y_yes)<(TR*(nk)):
            y_yes = np.append(int(y), y_yes)
        else: 
            y_excess = np.append(int(y), y_excess)
    
    np.random.shuffle(y_yes)
    np.random.shuffle(y_excess)
    y_y = y_yes.astype(int)
    print(" ** y_yes = ", len(y_yes), "y_excess = ", len(y_excess))
    y_excess = y_excess.astype(int)

    y_val = y_data[y_excess[0:vallim]]
    x_val = x_data[y_excess[0:vallim]]

    y_test = y_data[y_excess[vallim:int(len(y_excess))]]
    x_test = x_data[y_excess[vallim:int(len(y_excess))]]
    
    y_data = y_data[y_y]
    x_data = x_data[y_y]

    return [y_data, x_data, y_test, x_test, y_val, x_val]

def FScoreCalc(y_test, x_test, model):

    probsp = np.argmax(model.predict(x_test), axis=-1)
    y_test = np.argmax(y_test, axis =-1)

    f_1_score = sklearn.metrics.f1_score(y_test, probsp)
    f_001_score = sklearn.metrics.fbeta_score(y_test, probsp, beta=0.01)
    
    print('\n ** F1_Score: %s, F0.01_Score: %s' % (f_1_score, f_001_score))
    return [f_1_score, f_001_score]
