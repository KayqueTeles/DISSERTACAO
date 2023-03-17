# %% md

## Lens Challenge 2.0 - Classification

# %% md

### Define config

# %%

"""
Deep Bayesian strong lensing code

@author(s): Manuel Blanco Valentín (mbvalentin@cbpf.br)
            Clécio de Bom (clecio@debom.com.br)
            Brian Nord
            Jason Poh
            Luciana Dias
"""

""" Basic Modules """

###### Possible error
###### error OOM - reduction batch size and after limits gpus
import tensorflow as tf
import os, sys
sys.path.append('/home/kayque/LENSLOAD/CH2/')
data_folder = '/share/storage1/arcfinding/challenge2/'

"""
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
"""
import numpy as np

""" Execution time measuring """
from time import time
import time
import matplotlib
from utils import utils
""" keras backend to clear session """
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")
from astropy.io import fits
from keras.utils import Progbar

""" Matplotlib """
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

import cv2


# %% md

### Define chrono marks

# %%

def add_time_mark(chrono, label):
    tnow = time()
    chrono['marks'][0].append(label)
    chrono['marks'][1].append(tnow)
    chrono['elapsed'][0].append(label)
    if len(chrono['marks'][1]) > 1:
        telapsed = utils.ElapsedTime(chrono['marks'][1][-2])
    else:
        telapsed = utils.ElapsedTime(tnow)
    chrono['elapsed'][1].append(telapsed)


# %% md

### Initialize stuff

# %%

def normalizey(X, vmin=-4e-11, vmax=2.5e-10):
    X = np.clip(X, vmin, vmax)
    X = (X - vmin) / (vmax - vmin)
    X = np.log10(X.astype(np.float16) + 1e-10)
    mmin = X.min()
    mmax = X.max()
    X = (X - mmin) / (mmax - mmin)
    return X

def DataGeneratorCh2(num_samples, version, input_shape):
    print(" ** Using Manu's generator")
    foldtimer = time.perf_counter()

    # %% md

    ### Load data

    # %%

    data_dir = '/home/kayque/LENSLOAD/CH2/'
    catalog_name = 'image_catalog2.0train.csv'

    """ Load catalog before images """
    import pandas as pd

    catalog = pd.read_csv(os.path.join(data_dir, (data_folder+catalog_name)), header=28)  # 28 for old catalog

    """ Now load images using catalog's IDs """
    from skimage.transform import resize

    channels = ['H', 'J', 'Y']
    #channels = ['VIS', 'J', 'Y']
    #channels = ['VIS']
    #nsamples = len(catalog['ID'])
    print(' ** Using channels: ', channels)
    idxs2keep = []
    #missing_data = [13913, 26305, 33597, 44071, 59871, 61145, 70458, 88731, 94173]
    missing_data = [13912, 26304, 33596, 44070, 59870, 61144, 70457, 88730, 94172]
    for a in missing_data:
        labels = catalog['ID'].drop(a)
    labels = labels[0:num_samples]
    nsamples = len(labels)

    reload = False

    """ Try to load numpy file with images """
    if os.path.isfile(os.path.join(data_folder,'images.npy')) and not reload:  
        images = np.load(os.path.join(data_folder,'images.npy'))
        idxs2keep = list(np.load(os.path.join(data_folder,'idxs2keep.npy')))
    else:
        #Path('/share/storage1/arcfinding/challenge2/').parent
        #os.chdir('/share/storage1/arcfinding/challenge2/')
        #images = None
        images = np.zeros((len(labels), input_shape, input_shape, 3))
        """ Loop thru indexes """
        pbar = Progbar(nsamples - 1)
        for iid, cid in enumerate(labels):  # enumerate(labels):

            """ Loop thru channels"""
            for ich, ch in enumerate(channels):

                """ Init image dir and name """
                image_file = os.path.join(data_folder,
                                      'train',
                                      'Public',
                                      'EUC_' + ch,
                                      'imageEUC_{}-{}.fits'.format(ch, cid))

                if os.path.isfile(image_file):
                    #print(image_galsub_file)

                    #if os.path.isfile(image_file):# and os.path.isfile(image_galsub_file):

                    """ Import data with astropy """
                    image_data = fits.getdata(image_file, ext=0)
                    image_data = resize(image_data, (input_shape,input_shape))

                    """ Initialize images array in case we haven't done it yet """
                    if images is None:
                        images = np.zeros((nsamples, *image_data.shape, len(channels)))

                    """ Set data in array """
                    image_data[np.where(np.isnan(image_data))] = 0
                    #image_data = np.uint32(image_data)
                    #image_data = utils.imadjust(image_data)
                    #image_data = cv2.fastNlMeansDenoising(image_data, None, 30, 7, 21)
                    image_data = utils.center_crop_and_resize(image_data, 101)
                    images[iid,:,:,ich] = image_data
                    if iid not in idxs2keep:
                        idxs2keep.append(iid)
                else:
                    print('\tSkipping index: {} (ID: {})'.format(iid, cid))
                    break

            if iid % 1000 == 0 and iid != 0:
                pbar.update(iid)

        """ Now save to numpy file """
        #np.save(os.path.join(data_folder, 'images.npy'), images)
        #np.save(os.path.join(data_folder, 'idxs2keep.npy'), np.array(idxs2keep))

    apply_log = True
    images = utils.normalize(images, len(channels), apply_log=apply_log)
    print(' -- apply_log: ', apply_log)
    #count = 0
    #for tid in range(num_samples):
    #    for tch in range(len(channels)):
    #        image = images[tid, :, :, tch]
    #        print(image.shape)
    #        son = np.sum(image)
    #        print(son)
    #        sun = list(map(sum, image))
    #        sun2 = np.sum(sun)
    #        print(sun2)
    #        print(np.isnan(sun2))
    #        if np.isnan(sun2):
    #            print(' -- NaN detected!!')
    #            image = np.zeros((input_shape, input_shape))
    #            #image[np.where(np.isnan(image))] = 0
    #            images[tid, :, :, tch] = image
    #            count += 1
    #print(' -- NaN values converted: ', count)
    np.random.shuffle(idxs2keep)
    #images = images.astype(np.float16)
    catalog = catalog.loc[idxs2keep]
    images = images[idxs2keep]
    #images = images[:num_samples]
    catalog = catalog[:num_samples]
    print(len(images), len(catalog))

    fig = plt.figure(figsize=(10,3))
    NN = images.shape[-1]
    for i in range(NN):
        plt.subplot(1,NN,i+1)
    _ = plt.hist(np.clip(images[:,:,:,i].flatten(),-0.4e-10,2.5e-10),bins=256)
    _ = plt.title(channels[i])
    fig.savefig('histogram_for_normalization.png')

    is_lens = (catalog['n_source_im'] > 0) & (catalog['mag_eff'] > 1.6) & (catalog['n_pix_source'] > 20)  # 700
    is_lens = 1.0 * is_lens
    #is_lens = to_categorical(is_lens, 2)
    print(catalog['ID'])
    print(is_lens)

    inputs = images#.astype(np.float16) 
    outputs = is_lens.to_numpy()
    
    #images = utils.normalize(images)
    print(inputs[0,:,:,:])
    #print(outputs)

    index = utils.save_clue(inputs, outputs, num_samples, version, 'generator', input_shape, 8, 8, 0, channels)
    #inputs = {'images': inputs}
    #outputs = {'is_lens': outputs}
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Data Generation TIME: %.3f minutes.' % elaps)
    return (inputs, outputs, index, channels)

