import warnings
import tensorflow as tf
from pathlib import Path
import os
import numpy as np
import shutil
import csv
import cv2
import time
import h5py
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from keras import backend as k_back
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from tensorflow.keras import backend as K
import numpy as np
from IPython.display import Image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_auc_score
from PIL import Image
import bisect
from opt import RAdam
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2

from utils import files_changer, utils, graphs
import model_lib
from icecream import ic

warnings.filterwarnings('ignore')

ic(' THIS CODE IS SUPPOSED TO WORK WITH TENSORFLOW 2.0+, KERAS 2.3+')
ic(tf.__version__) ## Tensorflow version of your current environment:'
ic(tf.test.is_gpu_available())  ## Is any GPU available for our code?'

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

###############################################################################################
#DEFINE CODE'S PARAMETERS
###############################################################################################

version = 101  # EXECUTION VERSION
num_classes = 2  # NUMBER OF OUTPUT CLASSES
rede = 'ensemble'     ##OPTIONS: 'resnet', 'ensemble' or 'effnet'
weights = None   #'imagenet' or None
preload_weights = False
testing = False
ch2_testing = False
ch1_testing = False
ch1_weights = False
aug_data = False     ##DO YOU WANT SOME DATA AUGMENTATION?
aug_type = ['rotation_range=90, horizontal_flip=True, vertical_flip=True']
learning_rate = 0.01
optimizer = 'sgd' #'sgd'# or 'adam' or 'nadam'
avgpool = True
dropout = False
loss = 'binary_crossentropy'  # 'categorical_crossentropy' #'fbeta'  #'binary_crossentropy' usually
loss_regularization = False   ###THIS APPARENTLY INCREASES THE VALIDATION LOSS
num_epochs = 50  # NUMBER OF EPOCHS
batch_size = 128  # Tamanho do Batch
k_folds = 5  # NÚMERO DE FOLDS
percent_data = 1.0  # Porcentagem de Dados DE TESTE a ser usado - TODO - Definir como será utilizado
vallim = 3000  # Quantidade de dados de validação
challenge = 'challenge1'
##about types of ensembles used
dirichlet = False
logistic = False
dataset_size = 20000
input_shape = 101
classes = ['lens', 'not-lens']
model_list = ['resnet50', 'effnet_B2']
models_list = model_list
#model_list = ['effnet_B2', 'resnet50']

###############################################################################################
#==============================================================================================
###############################################################################################

ic(' ** Going to loop through models: ', model_list)
l_ml = len(model_list)
l_ml1 = len(model_list)+1

if challenge == 'challenge1':
    version = 'C_101'   ##C1 is being used to massive trainings
    optimizer = 'sgd'
    dataset_size = 20000
    vallim = 2000
    loss = 'binary_crossentropy'

if testing:
    version = 'T3'  # VERSÃO PRA COLOCAR NAS PASTAS
    vallim = 50
    num_epochs = 5
    percent_data = 0.5
    dataset_size = 2000

if ch2_testing:
    version = 'B1'

ic(" ** Are we performing tests? :", testing)
ic(" ** Chosen parameters:")
code_data =[["learning rate:", learning_rate],
            ["classes:", num_classes],
            ["input_shape:", input_shape],
            ["augmented:", aug_data],["avg_pool?", avgpool],["loss:", loss],
            ["dropout:", dropout],
            ["preloading weights? ", preload_weights], 
            ["dataset_size:", dataset_size], ["valid:", vallim],
            ["percent_data:", percent_data], ["batch_size:", batch_size],
            ["num_epochs:", num_epochs], ["k_folds:", k_folds],
            ["weights:", weights],["testing?", testing],
            ['loss regularization?', loss_regularization],
            ["ch1_testing?", ch1_testing],["dirich_ensemble?", dirichlet],
            ["logistic_ensemble?", logistic],
            ['challenge:', challenge], ["VERSION:", version]]
ic(code_data)

with open('code_parameters_version_%s.csv' % version, 'w', newline='') as g:
    writer = csv.writer(g)
    writer.writerow(code_data)  #htop

if challenge == 'challenge1':
    #train_data_sizes = [16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2000.0, 1750.0, 1500.0, 1250.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 480.0, 460.0, 440.0, 420.0, 500.0, 380.0, 360.0, 340.0, 320.0, 300.0, 180.0, 160.0, 140.0, 120.0, 100.0, 200.0, 280.0, 260.0, 240.0, 220.0, 400.0, 100, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
    #train_data_sizes = [200.0, 2500, 2000.0, 1750.0, 1500.0, 1250.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]
    train_data_sizes = [480.0, 460.0, 440.0]
    #train_data_sizes = [16000.0]

if testing:
    train_data_sizes = [100.0, 200.0]
ic(train_data_sizes)
np.random.shuffle(train_data_sizes)

########################################################
# CHECKING PHYSICAL DEVICES
ic(len(tf.config.experimental.list_physical_devices('GPU')))    ## Num GPUs Available: '
ic(len(tf.config.experimental.list_physical_devices('CPU')))    ## Num CPUs Available: ', 

### CHECKING DATASET
ic(' ** Verifying data...')
if challenge == 'challenge1':
    x_data_original, y_data_original = files_changer.data_downloader(dataset_size, k_folds, vallim, version)
elif challenge == 'challenge2':
    #from data_generator_challenge2 import DataGenerator
    from datagen_chal2 import DataGeneratorCh2
    #x_data_original, y_data_original, index, channels = DataGenerator(dataset_size, version, input_shape)
    x_data_original, y_data_original, index, channels = DataGeneratorCh2(dataset_size, version, input_shape)
elif challenge == 'both':
    x_data_1, y_data_1, index, channels = files_changer.data_downloader(dataset_size, version, input_shape)
    from datagen_chal2 import DataGeneratorCh2
    x_data_2, y_data_2, index, channels = DataGeneratorCh2(dataset_size, version, input_shape)

if ch2_testing:
    from datagen_chal2 import DataGeneratorCh2
    x_test_original, y_test_original, index, channels = DataGeneratorCh2(10000, version, input_shape)


Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

# Contador de tempo Total
begin = time.perf_counter()
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

##########################################33
# Apangando dados de testes anteriores para executar o novo teste
ic(' ** Cleaning up previous files...')

utils.create_path('ENSEMBLE_%s' % version)

ic(os.getcwd())
types = ['final', 'best', 'train']

for u in range(0, len(train_data_sizes)):
    begin_fold = time.perf_counter()
    train_size = train_data_sizes[u]
    ic(' ** NEW CICLE WITH %s TRAINING SAMPLES! **************************************************************************************************' % train_size)
    ic(' ** Cleaning up previous files and folders...')
    files_changer.file_remover(train_size, k_folds, version, model_list, num_epochs)

    ic(' ** Starting data preprocessing...')
    y_data = y_data_original
    x_data = x_data_original

    utils.create_path('ENSEMBLE_%s/' % (version), inside=True, train_size=train_size)

    #ic(' ** y_data shape: ', y_data.shape, ' ** Total dataset size: ', len(y_data), 'objects.')
    #ic(' ** Balancing number of samples on each class for train+val sets with %s samples...' % train_size)
    y_data, x_data, y_test, x_test, y_val, x_val = utils.test_samples_balancer(y_data, x_data, vallim, train_size, percent_data, challenge)
    if ch2_testing:
        x_test = x_test_original
        y_test = y_test_original

    y_test = to_categorical(y_test, num_classes = 2)
    if train_size < 1000:
        x_test = x_test[0:int(len(y_test)*0.5),:,:,:]
        y_test = y_test[0:int(len(y_test)*0.5)]

    #from utils.dnn import split_data
    #x_data, y_data, x_val, y_val, x_test, y_test, idxs = split_data(x_data, y_data, validation_split = .1)

    ic(' ** y_data arranged with format:  ** y_test:   ', y_test.shape, ' ** y_data:  ', y_data.shape, ' ** y_val:  ', y_val.shape)
    ic(' ** x_data splitted with format:  ** x_test:   ', x_test.shape, ' ** x_data:  ', x_data.shape, ' ** x_val:  ', x_val.shape)

    ic(' ** Converting data and list of indices into folds for cross-validation...')

    subset_size = int(len(y_data) / k_folds)
    folds = utils.load_data_kfold(k_folds, x_data, y_data)

    ic(' ** Starting network training... \n')

    start = time.perf_counter()
    roc_m = np.zeros((l_ml1,4,k_folds), dtype = object)  ###0: AUC; 1:FPR; 2:TPR; 3:AUC2. SETTING DTYPE ALLOWS ANYTHING TO BE PUT INSIDE THAT ARRAY
    auc_m = np.zeros((l_ml1,3), dtype = object)  ###0:HIGHAUC; 1:LOWAUC; 2:MEDAUC
    ic(roc_m.shape, auc_m.shape)
    roc_m_t = np.zeros((l_ml1,4,k_folds), dtype = object)  ###0: AUC; 1:FPR; 2:TPR; 3:AUC2. SETTING DTYPE ALLOWS ANYTHING TO BE PUT INSIDE THAT ARRAY
    auc_m_t = np.zeros((l_ml1,3), dtype = object)  ###0:HIGHAUC; 1:LOWAUC; 2:MEDAUC
    ic(roc_m_t.shape, auc_m_t.shape)
    roc_m_b = np.zeros((l_ml1,4,k_folds), dtype = object)  ###0: AUC; 1:FPR; 2:TPR; 3:AUC2. SETTING DTYPE ALLOWS ANYTHING TO BE PUT INSIDE THAT ARRAY
    auc_m_b = np.zeros((l_ml1,3), dtype = object)  ###0:HIGHAUC; 1:LOWAUC; 2:MEDAUC
    ic(roc_m_b.shape, auc_m_b.shape)

    #y_test = to_categorical(y_test, num_classes=2)
    #ic(" ** y_test: ", y_test)

    # Loop para a execução de todas as FOLDS
    for fold, (train_idx, val_idx) in enumerate(folds):

        ic(' **** New Fold ****')
        foldtimer = time.perf_counter()
        ic(' ** Fold: %s with %s training samples' % (fold, train_size))
        x_val_cv = x_val
        y_val_cv = y_val
        if train_size < 1600:
            ic(' ** Using Original Cross-Val method')
            x_data_cv = x_data[val_idx]
            y_data_cv= y_data[val_idx]
        else:
            ic(' ** Using Modified Cross-Val method')
            x_data_cv = x_data[train_idx]
            y_data_cv = y_data[train_idx]

        ic(' ** Converting vector classes to binary matrices...')
        y_data_cv_antes = y_data_cv
        y_val_cv_antes = y_val_cv
        y_data_cv = to_categorical(y_data_cv, num_classes=2)
        y_val_cv = to_categorical(y_val_cv, num_classes=2)

        # Data Augmentation
        if aug_data:
            # TODO - TENTANDO AUTOMATIZAR O TIPO DE AUGMENTATION
            ic(' ** Augmented Data: {}'. format(aug_type))
            gen = ImageDataGenerator(aug_type)
        else:
            ic(' ** Sem Augmented Data')
            gen = ImageDataGenerator()

        with tf.device('/GPU:0'):
            generator = gen.flow(x_data_cv, y_data_cv, batch_size=batch_size)
            ic(generator)
        
            ic(" ** Generating models...")
            models = []
            best_models = []
            ic(" ** from: ", models_list)
            cn = 0
            for ix in models_list:
                try:
                    for ty in types:
                        utils.create_path('/ENSEMBLE_%s/%s/%s_%s_%s_%s' % (version, train_size, ix, ty, version, train_size))
                    ic(' ** Current model: ', ix)
                    files_changer.file_remover(train_size, k_folds, version, model_list, num_epochs, only_weights=True)
                    model = utils.get_model_roulette(ix, x_data, weights)
                    ic(' ** Model Summary: \n', model.summary())
                    ic(' ** Training %s.' % ix)
                    callbacks = model_lib.compile_model(ix, model, optimizer, fold, version)
                    history = model_lib.fit_model(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks)
                    ic(' ** Training %s completed.' % model)
                    ic(' ** Plotting %s Graphs' % model)
                    graphs.accurary_graph(history, num_epochs, train_size, fold, ix, ix, version)
                    graphs.loss_graph(history, num_epochs, train_size, fold, ix, ix, version)
                    ###EVALUATING TEST DATA WITH FINAL MODEL
                    roc_m[cn,0,fold], roc_m[cn,1,fold], roc_m[cn,2,fold], thres = graphs.roc_graph(ix, model, x_test, y_test, ix, train_size, fold, version, ix, types[0])
                    #ic(roc_m[cn,0,fold], roc_m[cn,1,fold], roc_m[cn,2,fold], thres)
                    ###EVALUATING TRAIN DATA WITH FINAL MODEL
                    roc_m_t[cn,0,fold], roc_m_t[cn,1,fold], roc_m_t[cn,2,fold], thres = graphs.roc_graph(ix, model, x_data_cv, y_data_cv, ix, train_size, fold, version, ix, types[1])
                    #ic(roc_m_t[cn,0,fold], roc_m_t[cn,1,fold], roc_m_t[cn,2,fold], thres)
                    models.append(model)
                    ###EVALUATING TEST DATA WITH BEST MODEL
                    for pi in range(num_epochs):
                        try: 
                            model.load_weights('Train_model_weights_%s_%s_%s.h5' % (ix, pi, version))
                            ic('Best model found on epoch %s' % pi)
                        except:
                            try:
                                model.load_weights('Train_model_weights_%s_0%s_%s.h5' % (ix, pi, version))
                                ic('Best model found on epoch %s' % pi)
                            except:
                                ic('No best model found on epoch %s' % pi)
                    roc_m_b[cn,0,fold], roc_m_b[cn,1,fold], roc_m_b[cn,2,fold], thres = graphs.roc_graph(ix, model, x_test, y_test, ix, train_size, fold, version, ix, types[2])
                    #ic(roc_m_b[cn,0,fold], roc_m_b[cn,1,fold], roc_m_b[cn,2,fold], thres)
                    #scores = model.evaluate(x_test, y_test, verbose=0)
                    #ic(' ** %s - Large CNN Error: %.2f%%' % (ix, (100 - scores[1] * 100)))
                    cn = cn + 1
                    best_models.append(model)
                    ic(models, best_models)
                except:
                    ic('Moving on')

            ###ROC CURVES ENSEMBLES
            ic(' -- Calculating ensemble between groups.')
            roc_m[cn,0,fold], roc_m[cn,1,fold], roc_m[cn,2,fold], thres = graphs.roc_graphs_sec(rede, models, x_test, y_test, model_list, train_size, fold, version, 'ensemble', types[0])
            #ic(roc_m[cn,0,fold], roc_m[cn,1,fold], roc_m[cn,2,fold], thres)

            y_data_c = to_categorical(y_data, num_classes=2)

            roc_m_t[cn,0,fold], roc_m_t[cn,1,fold], roc_m_t[cn,2,fold], thres = graphs.roc_graphs_sec(rede, models, x_data, y_data_c, model_list, train_size, fold, version, 'ensemble_train', types[1])
            #ic(roc_m_t[cn,0,fold], roc_m_t[cn,1,fold], roc_m_t[cn,2,fold], thres)

            roc_m_b[cn,0,fold], roc_m_b[cn,1,fold], roc_m_b[cn,2,fold], thres = graphs.roc_graphs_sec(rede, best_models, x_test, y_test, model_list, train_size, fold, version, 'ensemble_best', types[2])
            #ic(roc_m_b[cn,0,fold], roc_m_b[cn,1,fold], roc_m_b[cn,2,fold], thres)

            elaps = (time.perf_counter() - foldtimer) / 60
            ic(' ** Fold TIME: %.3f minutes.' % elaps)
            K.clear_session() 

    # CLOSE Loop para a execução de todas as FOLDS
    ic(' ** Training and evaluation complete.')
    elapsed = (time.perf_counter() - start) / 60
    ic(' ** %.3f TIME: %.3f minutes.' % (train_size, elapsed))

    ic(" ** Generating code_data for models.")
    for ind in range(len(models)):
        auc_m[ind,0], auc_m[ind,1], auc_m[ind,2] = graphs.ultimate_ROC(roc_m[ind,2,:], thres, roc_m[ind,0,:], roc_m[ind,1,:], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        utils.roc_curves_sec(y_test, x_test, models[ind], model_list, version)
        ic(auc_m[ind,0], auc_m[ind,1], auc_m[ind,2])

        auc_m_t[ind,0], auc_m_t[ind,1], auc_m_t[ind,2] = graphs.ultimate_ROC(roc_m_t[ind,2,:], thres, roc_m_t[ind,0,:], roc_m_t[ind,1,:], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        utils.roc_curves_sec(y_data, x_data, models[ind], model_list, version)
        ic(auc_m_t[ind,0], auc_m_t[ind,1], auc_m_t[ind,2])

        auc_m_b[ind,0], auc_m_b[ind,1], auc_m_b[ind,2] = graphs.ultimate_ROC(roc_m_b[ind,2,:], thres, roc_m_b[ind,0,:], roc_m_b[ind,1,:], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        utils.roc_curves_sec(y_test, x_test, best_models[ind], model_list, version)
        ic(auc_m_b[ind,0], auc_m_b[ind,1], auc_m_b[ind,2])

        with open('./RESULTS/ENSEMBLE_%s/code_data_version_%s_model_%s_aucs.csv' % (version, version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, auc_m[ind,0], auc_m[ind,1], auc_m[ind,2]]
            writer.writerow(code_data)
        with open('./RESULTS/ENSEMBLE_%s/code_data_version_%s_model_%s_aucs_train.csv' % (version, version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, auc_m_t[ind,0], auc_m_t[ind,1], auc_m_t[ind,2]]
            writer.writerow(code_data)
        with open('./RESULTS/ENSEMBLE_%s/code_data_version_%s_model_%s_aucs_best.csv' % (version, version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, auc_m_b[ind,0], auc_m_b[ind,1], auc_m_b[ind,2]]
            writer.writerow(code_data)

    auc_m[l_ml,0], auc_m[l_ml,1], auc_m[l_ml,2] = graphs.ultimate_ROC(roc_m[(l_ml),2,:], thres, roc_m[(l_ml),0,:], roc_m[(l_ml),1,:], 'ensemble', 'ensemble', k_folds, train_size, 'ensemble', version)
    auc_m_t[l_ml,0], auc_m_t[l_ml,1], auc_m_t[l_ml,2] = graphs.ultimate_ROC(roc_m_t[(l_ml),2,:], thres, roc_m_t[(l_ml),0,:], roc_m_t[(l_ml),1,:], 'ensemble_train', 'ensemble_train', k_folds, train_size, 'ensemble_train', version)
    auc_m_b[l_ml,0], auc_m_b[l_ml,1], auc_m_b[l_ml,2] = graphs.ultimate_ROC(roc_m_b[(l_ml),2,:], thres, roc_m_b[(l_ml),0,:], roc_m_b[(l_ml),1,:], 'ensemble_best', 'ensemble_best', k_folds, train_size, 'ensemble_best', version)
    files_changer.filemover(train_size, version, k_folds, model_list, num_epochs)

    k_back.clear_session()

    time_fold = (time.perf_counter() - begin_fold) / (60 * 60)
    ic(' Ciclo ', u, ' concluido em: ', time_fold, ' horas.')

files_changer.last_mover(version, model_list, dataset_size, num_epochs, input_shape)

time_total = (time.perf_counter() - begin) / (60 * 60)
ic(' ** Mission accomplished in {} hours.'. format(time_total))
ic(' ** FINISHED! ************************')






