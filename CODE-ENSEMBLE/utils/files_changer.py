import os
import shutil
import tarfile
import zipfile
import wget
import numpy as np
import time
from icecream import ic
import pandas as pd
import h5py

path_folder = '/home/kayque/LENSLOAD/RESULTS'
# path_folder = 'C:/Users/Teletrabalho/Documents/Pessoal/resultados'
# path_folder = '/home/hdd2T/icaroDados'


# Método que faz o download do dataset, se necessário
def data_downloader(num_samples, n_folds, l_val, version):
    foldtimer = time.perf_counter()
    import wget, zipfile, tarfile

    PATH = '/home/kayque/LENSLOAD'
    var = PATH + '/' + 'lensdata/'

    if os.path.exists(var):
        ic(' creating dir')
    else:
        os.makedirs(var)

    ic(var + 'x_data20000fits.h5')
    if os.path.exists(var + 'x_data20000fits.h5'):
        ic(' ** Files from lensdata.tar.gz were already downloaded.')
    else:
        ic('\n ** Downloading lensdata.zip...')
        wget.download('https://clearskiesrbest.files.wordpress.com/2019/02/lensdata.zip', out=var)
        ic(' ** Download successful. Extracting...')
        with zipfile.ZipFile(var + 'lensdata.zip', 'r') as zip_ref:
            zip_ref.extractall(var)
            ic(' ** Extracted successfully.')
        ic(' ** Extracting data from lensdata.tar.gz...')
        tar = tarfile.open(var + 'lensdata.tar.gz', 'r:gz')
        tar.extractall(var)
        tar.close()
        ic(' ** Extracted successfully.')

    ic(var + 'x_data20000fits.h5')
    if os.path.exists(var + 'x_data20000fits.h5'):
        ic(' ** Files from lensdata.tar.gz were already extracted.')
    else:
        ic(' ** Extracting data from #DataVisualization.tar.gz...')
        tar = tarfile.open(var + 'DataVisualization.tar.gz', 'r:gz')
        tar.extractall(var)
        tar.close()
        ic(' ** Extracted successfully.')
        ic(' ** Extrating data from x_data20000fits.h5.tar.gz...')
        tar = tarfile.open(var + 'x_data20000fits.h5.tar.gz', 'r:gz')
        tar.extractall(var)
        tar.close()
        ic(' ** Extracted successfully.')
    if os.path.exists(var + 'lensdata.tar.gz'):
        os.remove(var + 'lensdata.tar.gz')
    if os.path.exists(var + 'lensdata.zip'):
        os.remove(var + 'lensdata.zip')

    for pa in range(0, 10, 1):
        if os.path.exists(var + 'lensdata ({}).zip'. format(pa)):
            os.remove(var + 'lensdata ({}).zip'. format(pa))
    ic('\n ** Starting data preprocessing...')
    labels = pd.read_csv(var + 'y_data20000fits.csv', delimiter=',', header=None)
    y_data = np.array(labels, np.uint8)
    y_data = y_data[:(num_samples*n_folds+l_val)]
    ic(len(y_data))

    x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')
    Ni_channels = 0  # first channel
    N_channels = 3  # number of channels
    channels = ['R', 'G', 'U']

    x_data = x_datasaved['data']
    x_data = x_data[:, :, :, Ni_channels:Ni_channels + N_channels]
    x_data = x_data[:(num_samples*n_folds+l_val), :, :, :]

    elaps = (time.perf_counter() - foldtimer) / 60
    ic(' ** Data Generation TIME: %.3f minutes.' % elaps)

    return (x_data, y_data)

def file_cleaner(k_folds, version, model_list, num_epochs, challenge_size):
    piccounter, weicounter, csvcounter = (0 for i in range(3))
    foldtimer = time.perf_counter()
    ic('\n ** Removing specified files and folders...')
    ic(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    m_list = model_list
    for mod in m_list:
        for train_size in range(challenge_size):
            for fold in range(0, 10 * k_folds, 1):
                if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
                if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                    os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                    counter = counter + 1
                # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
                if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                    os.remove('training_{}_fold_{}.csv'. format(mod, fold))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold)):
                    os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold))
                    counter = counter + 1  
                for epoch in range(0, num_epochs, 1):
                    epo = epoch + 1
                    if epoch < 10:
                    # Números menores que 10
                        if os.path.exists('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1
                    else:
                        # Números maiores que 10
                        if os.path.exists('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1

            # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
            if os.path.exists('{}_{}_{}/'. format(mod, version, train_size)):
                shutil.rmtree('{}_{}_{}/'. format(mod, version, train_size))

    ic('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    ic(' ** Removing TIME: %.3f minutes.' % elaps)


# Método que elimina arquivos de testes anteriores
def file_remover(train_size, k_folds, version, model_list, num_epochs, only_weights=False):
    piccounter, weicounter, csvcounter, counter = (0 for i in range(4))
    foldtimer = time.perf_counter()
    ic('\n ** Removing specified files and folders...')
    ic(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    m_list = model_list
    for mod in m_list:
        for fold in range(0, 10 * k_folds, 1):
            if only_weights:
                for epoch in range(0, num_epochs, 1):
                    epo = epoch + 1
                    if epoch < 10:
                    # Números menores que 10
                        if os.path.exists('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1
                    else:
                        # Números maiores que 10
                        if os.path.exists('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1
            else:
                if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
                if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                    os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                    counter = counter + 1
                # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
                if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                    os.remove('training_{}_fold_{}.csv'. format(mod, fold))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold)):
                    os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold))
                    counter = counter + 1  

    ic('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    ic(' ** Removing TIME: %.3f minutes.' % elaps)


def filemover(train_size, version, k_folds, model_list, num_epochs):
    # Checando se as pastas já existem, e apagando ou criando, de acordo com o caso
    ic('\n ** Moving created files to a certain folder.')
    foldtimer = time.perf_counter()
    counter = 0

    ic(" ** Checking if there's a results folder...")
    if os.path.exists('./RESULTS'):
        ic(' ** results file found. Moving forward.')
    else:
        ic(' ** None found. Creating one.')
        os.mkdir('./RESULTS')
        ic(' ** Done!')
    ic(" ** Checking if there's a local folder...")
    if os.path.exists('RESULTS/ENSEMBLE_%s' % version):
        ic(' ** Yes. There is.')
    else:
        ic(" ** None found. Creating one.")
        os.mkdir('RESULTS/ENSEMBLE_%s' % version)
        ic(" ** Done!")
    weicounter = 0
    m_list = model_list
    ic(m_list)

    for mod in m_list:
        ic(" ** Checking if there's an network({}) folder...". format(mod))
        if os.path.exists('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size)):
            # if os.path.exists('results/RNCV_%s_%s' % (version, train_size)):
            ic(' ** Yes. There is. Trying to delete and renew...')
            shutil.rmtree('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            # os.mkdir('results/RNCV_%s_%s' % (version, TR))
            ic(' ** Done!')
        else:
            ic(' ** None found. Creating one.')
            os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            ic(' ** Done!')
        
        dest1 = ('{}/ENSEMBLE_{}/{}_{}_{}/'. format(path_folder, version, mod, version, train_size))

        # Movendo os arquivos criados na execução do teste
        for fold in range(0, 10 * k_folds, 1):
            # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
            if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
            if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                shutil.move('./AUCxSize_{}_version_{}.png'. format(mod, version), dest1)
                counter = counter + 1
            # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
            if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                shutil.move('training_{}_fold_{}.csv'. format(mod, fold), dest1)
                counter = counter + 1
            if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            if mod == 'ensemble':
                if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold)):
                    shutil.move('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold))
                    counter = counter + 1  
            for epoch in range(0, num_epochs, 1):
                epo = epoch + 1
                if epoch < 10:
                # Números menores que 10
                    if os.path.exists('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version)):
                        os.remove('./Train_model_weights_{}_0{}_ver_{}.h5'. format(mod, epo, version))
                        weicounter = weicounter + 1
                else:
                    # Números maiores que 10
                    if os.path.exists('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version)):
                        os.remove('./Train_model_weights_{}_{}_ver_{}.h5'. format(mod, epo, version))
                        weicounter = weicounter + 1

    ic('\n ** Done. {} files moved, '. format(counter), '{} weights removed.'. format(weicounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    ic(' ** Moving TIME: %.3f minutes.' % elaps)

def last_mover(version, model_list):
    ic('\n ** Process is almost finished.')
    ic(' ** Proceeding to move LAST files to RESULTS folder.')
    counter = 0
    ic(" ** Checking if there's a RESULTS folder...")

    dest3 = ('/home/kayque/LENSLOAD/RESULTS/ENSEMBLE_%s' % version)
    ic(' ** Initiating last_mover...')
    m_list = model_list
    for mo in m_list:
        if os.path.exists('code_data_version_%s_model_%s_aucs.csv' % (version, m_list[mo])):
            shutil.move('code_data_version_%s_model_%s_aucs.csv' % (version, m_list[mo]), dest3)
            counter = counter + 1
        if os.path.exists('code_data_version_%s_model_%s_f1s.csv' % (version, m_list[mo])):
            shutil.move('code_data_version_%s_model_%s_f1s.csv' % (version, m_list[mo]), dest3)
            counter = counter + 1
    if os.path.exists('code_parameters_version_{}.csv'. format(version)):
        shutil.move('code_parameters_version_{}.csv'. format(version), dest3)
        counter = counter + 1

    ic(" ** Moving done. %s files moved." % counter)
