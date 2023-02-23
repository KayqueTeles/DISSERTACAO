import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pylab as py
import cv2
import scipy.misc
import warnings
import os
import keras
import time
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import tarfile
import wget
import zipfile

from astropy.io import fits
from keras import models
#from keras.models import Sequential
#from keras import layers
from keras.layers import Input
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Dense, Dropout, Flatten, Activation
#from keras.layers.normalization import BatchNormalization

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import plot_model
from IPython.display import Image, display
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from keras.utils import to_categorical
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.applications.resnet50 import ResNet50

warnings.filterwarnings("ignore")

print("\n ** Verifying data...")

if os.path.exists('./lensdata.zip'):
    print("\n ** Dataset lensdata.zip already downloaded.")
    if os.path.exists('lensdata.tar.gz'):
          print("\n ** lensdata.tar.gz already extracted from lensdata.tar.gz.")
    else:
          with zipfile.ZipFile("lensdata.zip", 'r') as zip_ref:
               zip_ref.extractall()
          print("\n ** Extracted successfully.")
else:
    print("\n ** Downloading lensdata.zip...")
    wget.download('https://clearskiesrbest.files.wordpress.com/2019/02/lensdata.zip')
    print("\n ** Download successful. Extracting...")
    with zipfile.ZipFile("lensdata.zip", 'r') as zip_ref:
         zip_ref.extractall() 
    print("\n ** Extracted successfully.")

if os.path.exists('./lensdata/DataVisualization.tar.gz'):
     print("\n ** Files from lensdata.tar.gz were already extracted.")
else:
     print("\n ** Extracting data from lensdata.tar.gz...")
     tar = tarfile.open("lensdata.tar.gz", "r:gz")
     tar.extractall()
     tar.close()
     print("\n ** Extracted successfully.")
if os.path.exists('./lensdata/x_data20000fits.h5'):
     print("\n ** Files from lensdata.tar.gz were already extracted.")
else:
     print("\n ** Extracting data from DataVisualization.tar.gz...")     
     tar = tarfile.open("./lensdata/DataVisualization.tar.gz", "r:gz")
     tar.extractall("./lensdata/")
     tar.close()
     print("\n ** Extracted successfully.")
     print("\n ** Extrating data from x_data20000fits.h5.tar.gz...")     
     tar = tarfile.open("./lensdata/x_data20000fits.h5.tar.gz", "r:gz")
     tar.extractall("./lensdata/")
     tar.close()
     print("\n ** Extracted successfully.")

print("\n ** Reading data from y_data20000fits.csv...")
#LOAD Y_DATA
PATH = os.getcwd()
var = PATH + "/" + "lensdata/"
y_batch = os.listdir(var)
labels = pd.read_csv(var + 'y_data20000fits.csv',delimiter=',', header=None)
#y_data = np.array(labels, np.uint8)

TR = 6000
y_data = np.array(labels, np.uint8)
y_size = len(y_data)
y_data.shape

print("\n **Randomizing y_data")
ind = np.arange(y_data.shape[0])
np.random.shuffle(ind)
y_data = y_data[ind]

print("\n ** Dataset has shape: ", y_data.shape)

print(" ** Total dataset size: ", y_size, "objects.")
print(" ** Starting y_data preprocessing...")

print(" ** Choosing testing and training samples ...")

y_test = y_data[0:2000,]
y_train = y_data[2000:(TR+2000),]
y_val = y_data[(TR+2000):y_size,]
#y_val = y_data[TR:y_size,]

print("\n ** y_data Formatted with shape:")
print(" ** y_val:   ", y_val.shape)
print(" ** y_test:  ", y_test.shape)
print(" ** y_train: ", y_train.shape)

print("\n ** Reading data from x_data20000fits.h5...")

import h5py
x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')

Ni_channels = 0 # first channnel
N_channels = 3 # number channels

x_data = x_datasaved['data']
x_size = len(x_data)

x_data = x_data[:,:,:,Ni_channels:Ni_channels + N_channels] 
print("\n ** Dataset has shape: ", x_data.shape)
print(" ** Total dataset size: ", x_size, "objects.")
print(" ** Applying randomized indices to x_data...")
x_data = x_data[ind]
print(" ** Choosing corresponding samples ...")

x_test = x_data[0:2000,:,:,:]
x_train = x_data[2000:(TR+2000),:,:,:]
x_val = x_data[(TR+2000):y_size,:,:,:]
#x_val = x_data[TR:x_size,:,:,:]

print(x_data.shape)

print("\n ** x_data Formatted with shape:")
print(" ** x_val:   ", x_val.shape)
print(" ** x_test:  ", x_test.shape)
print(" ** x_train: ", x_train.shape)

print("\n ** Building sequential model...")

K.set_image_data_format('channels_last')


img_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
img_input = Input(shape=img_shape)
if N_channels == 3:
   model = ResNet50(include_top=True, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
else:
   model = ResNet50(include_top=False, weights=None, input_tensor=img_input, input_shape=img_shape, classes=2, pooling=None)
    
model.save('ModelResnet50Lens.h5') # if save model
model.summary()

print("\n ** Compiling model...")

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss= 'binary_crossentropy' , optimizer='sgd' , metrics=[ 'accuracy' ])

print("\n ** Plotting model and callbacks...")
#gen = ImageDataGenerator(rotation_range = 90)
gen = ImageDataGenerator()

def get_callbacks(name_weights, patience_lr, name_csv):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True)
    csv_logger = CSVLogger(name_csv)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='max')
    return [mcp_save, csv_logger, reduce_lr_loss]

print("\n ** Converting vector classes to binary matrices...")
y_train = to_categorical(y_train,num_classes=2)

y_val = to_categorical(y_val,num_classes=2)

y_test = to_categorical(y_test,num_classes=2)

print("\n ** Starting network training... \n")

batch_size=64


#name_weights = "Train_model_weights_{epoch:02d}.h5" #if save weights epochs
name_weights = "Train_model_weights.h5" #if save last weigts
csv_name = "trainning_k.csv"
callbacks = get_callbacks(name_weights = name_weights, patience_lr=10, name_csv = csv_name)
generator = gen.flow(x_train, y_train, batch_size = batch_size)
# change epochs
history = model.fit_generator(
                  generator,
                  steps_per_epoch=len(x_train)/batch_size,
                  epochs=50,
                  verbose=1,
                  validation_data = (x_val, y_val),
                  validation_steps = len(x_val)/batch_size,
                  callbacks = callbacks)
      
      
print("\n ** Training completed.")
print("\n ** Plotting training & validation accuracy values.")

###### Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy ' )
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.savefig("AccxEpoch_%s.png" % TR)

print("\n ** Plotting training & validation loss values.")
###### Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss ')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.savefig("LossxEpoch_%s.png" % TR)

print("\n ** Model evaluation stage.")
#print(model.evaluate(x_test, y_test)) 
      
probs = model.predict(x_test)
######## keep probabilities for the positive outcome only
probsp = probs[:, 1]
######## calculate AUC
auc = roc_auc_score(y_test[:,1], probsp)
print('AUC: %.3f' % auc)

      
######## calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test[:,1], probsp)


plt.figure()
plt.plot([0, 1], [0, 1], 'k--') # k = color black
plt.plot(fpr, tpr, label="AUC: %.3f" % auc, color='C1', linewidth=3) # for color 'C'+str(j), for j[0 9]
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.title('ROC')
plt.xlabel('false positive rate', fontsize=14)
plt.ylabel('true positive rate', fontsize=14)
    
plt.show()
plt.savefig("ROCLensDetectNet_%s.png" % TR)

scores = model.evaluate(x_test, y_test, verbose=0)
print("\n ** Large CNN Error: %.2f%%" % (100-scores[1]*100))

print("\n ** FINISHED! ****")
