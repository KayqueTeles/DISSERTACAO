import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import time
import tensorflow as tf
import sklearn
from pathlib import Path
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
from keras.optimizers import SGD
from keras.optimizers import Adam
#from .threading import StoppableThread
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from IPython.display import Image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_auc_score
from PIL import Image
import bisect
#import model_lib
from joblib import Parallel, delayed
from skimage.transform import resize
from keras.utils import Progbar
from icecream import ic
import pandas as pd
import h5py
#from utils import graphs

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#K.tensorflow_backend.set_session(tf.Session(config=config))

def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    """
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:
    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array
    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
        .. versionadded:: 1.9.0
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
        .. versionadded:: 1.13.0
    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.
        .. versionadded:: 1.9.0
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    repeat : Repeat elements of an array.
    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.
    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    Return the unique rows of a 2D array
    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])
    Return the indices of the original array that give the unique values:
    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')
    Reconstruct the input array from the unique values and inverse:
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    Reconstruct the input values from the unique values and counts:
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved
    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim) from None

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x

def print_images_clecio_like(x_data, num_prints, num_channels, input_shape, index):
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')

    source = '/home/kayque/LENSLOAD/'
    ic(' ** Does an old folder exists?')
    if os.path.exists(source+'lens_yes_2'):
        ic(' ** Yes, it does! Trying to delete... ')
        shutil.rmtree(source+'lens_yes_2', ignore_errors=True)
        ic(" ** Supposedly done. Checking if there's an RNCV folder...")
        os.mkdir('lens_yes_2')
        ic(' ** Done!')
    else:
        ic(" ** None found. Creating one.")
        os.mkdir('lens_yes_2')
        ic(' ** Done!')

    dest1 = ('/home/kayque/LENSLOAD/lens_yes_2/')
    counter = 0

    for sm in range(num_prints):
        img_imdjust = np.zeros((input_shape, input_shape, num_channels))
        img_imdj_n_denoi = np.zeros((input_shape, input_shape, num_channels))
        for ch in range(num_channels):
            img_ch = x_data[sm,:,:,ch]
            img_ch = np.uint8(cv2.normalize(np.float32(img_ch), None, 0, 255, cv2.NORM_MINMAX))
            img_chi = imadjust(img_ch)
            img_ch = cv2.fastNlMeansDenoising(img_chi, None, 30, 7, 21)
            img_imdjust[:,:,ch] = img_chi
            img_imdj_n_denoi[:,:,ch] = img_ch
        rgb = toimage(img_imdj_n_denoi)
        rgb = np.array(rgb)
        rgbi = toimage(img_imdjust)
        rgbi = np.array(rgbi) 

        plt.figure(1)
        plt.subplot(141)
        plt.imshow(img_imdjust[:,:,0], cmap='gray')
        plt.title('Band H')
        plt.grid(False)
        plt.subplot(142)
        plt.imshow(img_imdjust[:,:,1], cmap='gray')
        plt.title('Band J')
        plt.grid(False)
        plt.subplot(143)
        plt.imshow(img_imdjust[:,:,2], cmap='gray')
        plt.title('Band Y')
        plt.grid(False)
        plt.subplot(144)
        plt.imshow(rgbi)
        plt.title('Result RGB')
        plt.grid(False)
        plt.savefig('img_I_{}_{}.png'.format(num_prints, index))

        plt.figure(2)
        plt.subplot(141)
        plt.imshow(img_imdj_n_denoi[:,:,0], cmap='gray')
        plt.title('Band H')
        plt.grid(False)
        plt.subplot(142)
        plt.imshow(img_imdj_n_denoi[:,:,1], cmap='gray')
        plt.title('Band J')
        plt.grid(False)
        plt.subplot(143)
        plt.imshow(img_imdj_n_denoi[:,:,2], cmap='gray')
        plt.title('Band Y')
        plt.grid(False)
        plt.subplot(144)
        plt.imshow(rgb)
        plt.title('Result RGB')
        plt.grid(False)
        plt.savefig('./graphs/img_I_D_{}_{}.png'.format(num_prints, index))

    ic("\n ** Done. %s files moved." % counter)
    return index

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.
    Return the sorted, unique values that are in both of the input arrays.
    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.
        .. versionadded:: 1.15.0
    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Examples
    --------
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])
    To intersect more than two arrays, use functools.reduce:
    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])
    To return the indices of the values common to the input arrays
    along with the intersected values:
    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))
    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d

def imadjust(src, tol=0.5, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if (vin[1] - vin[0]) > 0:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    else:
        scale = 0
        
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    #####CRIAR BATCHES COM POTÊNCIAS DE 2 PARA RESOLVER O PROBLEMA DE 450 SAMPLES DA
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

#*******************************************************


def save_clue(x_data, TR, version, step, input_shape, nrows, ncols, index, vis=False):

    figcount = index
    plt.figure()
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20))
    for i in range(nrows):
        for j in range(ncols):
            if not vis:
                temp_image = toimage(np.array(x_data[figcount, :, :, :]))
            else:
                temp_image = toimage(np.array(np.concatenate([x_data[:,:,:,0:],x_data[:,:,:,0:],x_data[:,:,:,0:]], axis=-1)))
            axs[i, j].imshow(temp_image)
            #axs[i, j].set_title('Class: %s' % y_data[figcount])
            figcount = figcount + 1
    if figcount > x_data.shape[0]:
        index = 1
    else:
        index = figcount + 1
    plt.savefig("./graphs/graphs_{}/ver_{}/CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png". format(TR, version, TR, version, step, input_shape, input_shape, index))
    ic("CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png saved.". format(TR, version, step, input_shape, input_shape, index))

    #index = print_images_clecio_like(x_data=x_data, num_prints=10, num_channels=len(channels), input_shape=input_shape, index)
    return index

def print_images_c(TR, x_data, num_prints, num_channels, version, input_shape, step, index):
    counter = 0

    plt.figure()
    #plt.subplots(num_prints+1, 8, 8)
    fig, axs = plt.subplots(num_prints, 4, figsize=(20,20))

    for sm in range(num_prints):
        img_imdjust = np.zeros((input_shape, input_shape, 3))
        img_imdj_n_denoi = np.zeros((input_shape, input_shape, 3))
        for ch in range(num_channels):
            img_ch = x_data[sm,:,:,ch]
            img_ch = np.uint8(cv2.normalize(np.float32(img_ch), None, 0, 255, cv2.NORM_MINMAX))
            img_chi = imadjust(img_ch)
            img_ch = cv2.fastNlMeansDenoising(img_chi, None, 30, 7, 21)
            img_imdjust[:,:,ch] = img_chi
            img_imdj_n_denoi[:,:,ch] = img_ch
        rgb = toimage(img_imdj_n_denoi)
        rgb = np.array(rgb)
        rgbi = toimage(img_imdjust)
        rgbi = np.array(rgbi) 

        #sm8 = (sm*8)
        #plt.subplot(num_prints, 8, sm+1)
        #plt.imshow(img_imdjust[:,:,0], cmap='gray')
        axs[sm, 0].imshow(img_imdjust[:,:,0])
        if sm == 1:
            axs[sm, 1].set_title('Band 1')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+2)
        #plt.imshow(img_imdjust[:,:,1], cmap='gray')
        axs[sm, 1].imshow(img_imdjust[:,:,1])
        if sm == 1:
            axs[sm, 1].set_title('Band 2')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+3)
        #plt.imshow(img_imdjust[:,:,2], cmap='gray')
        axs[sm, 2].imshow(img_imdjust[:,:,2])
        if sm == 1:
            axs[sm, 1].set_title('Band 3')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+4)
        #plt.imshow(rgbi)
        axs[sm, 3].imshow(rgbi)
        if sm == 1:
            axs[sm, 1].set_title('Result')
        #plt.grid(False)
        #plt.savefig('./graphs/img_I_{}_{}_ver_{}_{}.png'.format(num_prints, step, version, index))

        #plt.subplot(num_prints, 8, sm+5)
        #plt.imshow(img_imdj_n_denoi[:,:,0], cmap='gray')
        #axs[sm, 5].imshow(img_imdjust[:,:,0])
        #plt.title('Band H')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+6)
        #plt.imshow(img_imdj_n_denoi[:,:,1], cmap='gray')
        #plt.title('Band J')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+7)
        #plt.imshow(img_imdj_n_denoi[:,:,2], cmap='gray')
        #plt.title('Band Y')
        #plt.grid(False)
        #plt.subplot(num_prints, 8, sm+8)
        #plt.imshow(rgb)
        #plt.title('Result HJY')
        #plt.grid(False)

    plt.savefig('./graphs/graphs_%s/ver_%s/imgs_step_%s_v_%s_%s.png' % (TR, version, step, version, index))


    ic(" ** Done. Files Stacked.")
    return index

def print_multi(TR, x_data, x_data_vis, num_prints, num_channels, version, input_shape, step, index):
    counter = 0

    plt.figure()
    #fig, axs = plt.subplots(num_prints, 5, figsize=(10,10))
    fig, axs = plt.subplots(num_prints, 5)

    for sm in range(num_prints):
        img_imdjust = np.zeros((input_shape, input_shape, 3))
        #img_imdj_n_denoi = np.zeros((input_shape, input_shape, 3))
        for ch in range(num_channels):
            img_ch = x_data[sm,:,:,ch]
            img_ch = np.uint8(cv2.normalize(np.float32(img_ch), None, 0, 255, cv2.NORM_MINMAX))
            img_chi = imadjust(img_ch)
            #img_ch = cv2.fastNlMeansDenoising(img_chi, None, 30, 7, 21)
            img_imdjust[:,:,ch] = img_chi
            #img_imdj_n_denoi[:,:,ch] = img_ch
        #rgb = toimage(img_imdj_n_denoi)
        #rgb = np.array(rgb)
        rgbi = toimage(img_imdjust)
        rgbi = np.array(rgbi) 
        tmp = x_data_vis[sm,:,:,0]
        #tmp = cv2.resize(tmp, (66,66))
        #tmp = tmp.reshape(66,66,1)
        tmp = np.uint8(cv2.normalize(np.float32(tmp), None, 0, 255, cv2.NORM_MINMAX))
        tmp = imadjust(tmp)

        axs[sm, 0].imshow(img_imdjust[:,:,0])
        if sm == 1:
            axs[sm, 0].set_title('Band H')
        axs[sm, 1].imshow(img_imdjust[:,:,1])
        if sm == 1:
            axs[sm, 1].set_title('Band J')
        axs[sm, 2].imshow(img_imdjust[:,:,2])
        if sm == 1:
            axs[sm, 2].set_title('Band Y')
        axs[sm, 3].imshow(rgbi)
        if sm == 1:
            axs[sm, 3].set_title('Result HJY')
        axs[sm, 4].imshow(tmp)
        if sm == 1:
            axs[sm, 4].set_title('Band VIS')

    plt.savefig('./graphs/graphs_%s/ver_%s/imgs_step_%s_v_%s_%s.png' % (TR, version, step, version, index))


    ic(" ** Done. Files Stacked.")
    return index
    
def histo(X, before_norm, bands, vis=False):
    ic(X.shape, before_norm, bands)
    for i in range(X.shape[3]):
        band = bands[i]
        ic(i, band, bands)
        a = X[:,:,:,i]
        vmin = -np.percentile(-a, 99.9)
        vmax = np.percentile(a, 98)
        ic(a.shape)
        plt.figure()
        plt.hist(a.ravel(), bins = np.linspace(vmin, vmax, 100))
        plt.xlim([0.0, 1.05])
        if before_norm:
            plt.title('Pixel Histogram for %s band.' % (band))
            plt.savefig('./graphs/HIST_for_band_%s_before_split_%s_vis_%s.png' % (band, before_norm, vis))
        else:
            plt.title('Pixel Histogram for %s band.' % (band))
            plt.savefig('./graphs/HIST_for_band_%s_after_split_%s_vis_%s.png' % (band, before_norm, vis))

def histo_all(X, Y, before_norm, bands, vis=False):
    ic(X.shape, before_norm, bands)
    plt.figure()
    for i in range(X.shape[3]):
        band = bands[i]
        ic(i, band, bands)
        a = X[:,:,:,i]
        vmin = -np.percentile(-a, 99.9)
        vmax = np.percentile(a, 98)
        ic(a.shape)
        plt.hist(a.ravel(), bins = np.linspace(vmin, vmax, 100), label='%s' % band, alpha = 0.2)
        #plt.xlim([0.0, 1.05])
    if vis:
        a = Y[:,:,:,0]
        vmin = -np.percentile(-a, 99.9)
        vmax = np.percentile(a, 98)
        ic(a.shape)
        plt.hist(a.ravel(), bins = np.linspace(vmin, vmax, 100), label='VIS', alpha = 0.2)
    if before_norm:
        plt.title('Pixel Histogram for All bands.')
        plt.savefig('./graphs/HIST_ALL_bn_%s.png' % (before_norm))
    else:
        plt.title('Pixel Histogram for All bands.')
        plt.savefig('./graphs/HIST_ALL_an_%s.png' % (before_norm))
    ic(' -- histo all saved.')

def load_both_data(data_dir, num_samples, version, vis=False, band=0):
    if vis:
        images_vis = np.load(os.path.join(data_dir,'images_vis_new.npy'))
        #images_vis = np.memmap((os.path.join(data_dir, 'images_vis_kay_20000_2.npy')), dtype='float32', mode='r+', shape=(99991, 200, 200, 3))
        histo(images_vis, True, ['VIS01'])
    else:
        images_vis = images_vis = np.load(os.path.join(data_dir,'images_hjy_new.npy'))
        #images_vis = np.memmap((os.path.join(data_dir, 'images_hjy_kay_20000_2.npy')), dtype='float32', mode='r+', shape=(19999, 66, 66, 3))
        histo(images_vis, True, ['H01', 'H02', 'H03'])
    ic(images_vis.shape)
    #index = save_clue(images_vis, num_samples, version, ('vis_lbd_%s_kay') % vis, images_vis.shape[1], 5, 5, 0, vis=vis)
    #images_vis = img_conv_3(images_vis)
    #index = print_images_c(images_vis, num_prints=5, num_channels=images_vis.shape[3], version=version, input_shape=images_vis.shape[1], step=('vis_lbd_%s_kay') % vis, index=0)

    images_vis = images_vis[:,:,:,band]
    #images_vis = images_vis[0:99991,:,:]
    #images_vis = images_vis[0:num_samples,:,:]
    
    ic(images_vis.shape)
    a = images_vis[:,:,:]
    #vmax = np.max(a)
    #vmax = np.percentile(a, 98)
    #vmin = -np.percentile(-a, 99.9)
    ic(a.shape)#, vmax, vmin)
    if vis:
        images_vis_2 = np.load(os.path.join(data_dir, 'images_vis_normalized.npy'))
        histo(images_vis_2, False, ['VIS01'])
    else:
        images_vis_2 = np.load(os.path.join(data_dir, 'images_hjy_normalized.npy'))
        histo(images_vis_2, False, ['H01', 'H02', 'H03'])
    ic(images_vis_2.shape)
    #index = save_clue(images_vis_2, num_samples, version, ('vis_lbd_%s_pat') % vis, images_vis_2.shape[1], 5, 5, 0, vis=vis)
    #images_vis_2 = img_conv_3(images_vis_2)
    images_vis_2 = images_vis_2[:,:,:,band]
    #images_vis_2 = images_vis_2[0:99991,:,:]
    #images_vis_2 = images_vis_2[0:num_samples,:,:]
    ic(images_vis_2.shape)

    sub = (np.sum(images_vis-images_vis_2))
    ic(sub)
    del sub
    b = images_vis_2[:,:,:]
    #index = print_images_c(images_vis_2, num_prints=5, num_channels=images_vis_2.shape[3], version=version, input_shape=images_vis_2.shape[1], step=('vis_lbd_%s_pat') % vis, index=0)
    del images_vis
    del images_vis_2
    ic(b.shape)#,vmax, vmin)
    plt.figure()
    vmax = np.percentile(a, 98)
    vmin = -np.percentile(-a, 99.9)
    ic(a.shape,vmax, vmin)
    plt.hist(a.ravel(), bins = np.linspace(vmin, vmax, 100), alpha = 0.2)
    vmax = np.percentile(b, 98)
    vmin = -np.percentile(-b, 99.9)
    ic(b.shape,vmax, vmin)
    plt.hist(b.ravel(), bins = np.linspace(vmin, vmax, 100))
    del a
    del b
    #plt.xlim([0.0, 1.05])
    plt.savefig('./graphs/HIST_for_vis_%s_before_norm_band_%s.png' % (vis, band))
    ic(' histo saved.')

def img_conv_3(images_vis):
    ic(' -- converting vis imgs')
    img_z = np.zeros((images_vis.shape[0],images_vis.shape[1],images_vis.shape[2],3), dtype="float32")
    img_z[:,:,:,0] = images_vis[:,:,:,0]
    images_vis_b = img_z
    del img_z
    ic(' -- done. ')
    ic(images_vis_b.shape)
    return images_vis_b

def create_path(study, version):
    root_dir = '/home/kayque/LENSLOAD/SCRIPTS/challenge2-main/graphs'
    output_directory = root_dir+'/graphs_%s/' % (study)
    if os.path.exists(output_directory):
        ic(' creating dir')
    else:
        os.makedirs(output_directory)
    output_directory = root_dir+'/graphs_%s/ver_%s/' % (study, version)
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

def load_data_kfold(k, x_data, y_data):
    ic('Preparing Folds')
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_data, y_data))
    return folds

def ultimate_ROC(lauc, lauc2, thres, tpr_all, tpr_all2, fpr_all, fpr_all2, name_file_rede, title_graph_rede, k_folds, train_size, rede, version):
    ic('\n ** Generating ultimate ROC graph for %s...' % rede)
    medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

    mauc = np.percentile(lauc, 50.0)
    #mAUCall = np.percentile(auc_all, 50.0)
    ic(" ** len tpr_all:", len(tpr_all[0]))
    ic(" ** len fpr_all:", len(fpr_all[0]))
    ic(" ** lauc:", len(lauc))
    ic(' -- thres: ', len(thres))
    
    #ic(" ** auc_all:", auc_all)

    for num in range(0, len(tpr_all), 1):
        lis = [item[num] for item in tpr_all]
        los = [item[num] for item in fpr_all]

        medians_x.append(np.percentile(los, 50.0))
        medians_y.append(np.percentile(lis, 50.0))
        lowlim.append(np.percentile(lis, 15.87))
        highlim.append(np.percentile(lis, 84.13))

    #lowauc = metrics.auc(medians_x, lowlim)
    #highauc = metrics.auc(medians_x, highlim)
    lowauc = np.percentile(lauc, 15.87)
    highauc = np.percentile(lauc, 84.13)
    ic('\n\n\n ** IS THIS CORRECT?')
    ic(lowauc, mauc, highauc)
    with open('code_data_version_%s_aucs_0_%s.csv' % (version, train_size), 'a', newline='') as f:
        writer = csv.writer(f)
        code_data = [train_size, lowauc, mauc, highauc]
        writer.writerow(code_data)
    ic(medians_x, medians_y, lowlim, highlim)
    ic(len(medians_x), len(medians_y), len(lowlim), len(highlim))
    #ic(lowauc, mAUCall, highauc)

    medians2_y, medians2_x, lowlim2, highlim2 = ([] for i in range(4))

    mauc2 = np.percentile(lauc2, 50.0)
    #mAUCall = np.percentile(auc_all, 50.0)
    ic(len(tpr_all2))
    ic(len(fpr_all2))
    ic(len(lauc2))
    ic(len(thres))
    #ic(" ** auc_all:", auc_all)

    for num in range(0, len(tpr_all2), 1):
        lis = [item[num] for item in tpr_all2]
        los = [item[num] for item in fpr_all2]

        medians2_x.append(np.percentile(los, 50.0))
        medians2_y.append(np.percentile(lis, 50.0))
        lowlim2.append(np.percentile(lis, 15.87))
        highlim2.append(np.percentile(lis, 84.13))

    #lowauc = metrics.auc(medians_x, lowlim)
    #highauc = metrics.auc(medians_x, highlim)
    lowauc2 = np.percentile(lauc2, 15.87)
    highauc2 = np.percentile(lauc2, 84.13)
    ic('\n\n\n ** IS THIS CORRECT?')
    ic(lowauc2, mauc2, highauc2)
    with open('code_data_version_%s_aucs_1_%s.csv' % (version, train_size), 'a', newline='') as f:
        writer = csv.writer(f)
        code_data = [train_size, lowauc2, mauc2, highauc2]
        writer.writerow(code_data)
    ic(medians2_x, medians2_y, lowlim2, highlim2)
    ic(len(medians2_x), len(medians2_y), len(lowlim2), len(highlim2))
    #ic(lowauc, mAUCall, highauc)

    # Plotting Final ROC graph
    final_roc_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, medians2_x, medians2_y, mauc2, lowlim2, highlim2, name_file_rede, title_graph_rede, version)

    return highauc, lowauc, mauc

def final_roc_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, medians_x2, medians_y2, mauc2, lowlim2, highlim2, name_file_rede, title_graph_rede, version):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.title('Median ROC {} over {} folds with {} training samples'.format(title_graph_rede, k_folds, train_size))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.plot(medians_x, medians_y, 'b', label='AUC: {}'.format(mauc), linewidth=3)
    plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
    plt.plot(medians_x2, medians_y2, 'r', label='AUC: {}'.format(mauc), linewidth=3)
    plt.fill_between(medians_x2, medians_y2, lowlim2, color='red', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x2, highlim2, medians_y2, color='red', alpha=0.3, interpolate=True)
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.savefig('./graphs/graphs_{}/ROCLensDetectNet_{}_Full_{}_version_{}.png'.format(train_size, name_file_rede, train_size, version))

def roc_curve_calculate(y_test, x_test, probs, rede):
    ic(' ** Roc Curve Calculating')
    ic('\n ** Preds for: ', rede)
    
    ic(len(probs))

    probsp = probs
    y_new = y_test
    ic(len(probs), len(y_new))
    thres = int(len(y_test))

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))

    for tt in range(len(threshold_v)):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(len(probsp)):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:
                    tp_score = tp_score + 1
                else:
                    fp_score = fp_score + 1
            else:
                if y_new[xz] == 0:
                    tn_score = tn_score + 1
                else:
                    fn_score = fn_score + 1
        tp_rate = tp_score / (tp_score + fn_score)
        fp_rate = fp_score / (fp_score + tn_score)
        tpr.append(tp_rate)
        fpr.append(fp_rate)

    #auc2 = roc_auc_score(y_test[:, 1], probsp)
    auc = metrics.auc(fpr, tpr)

    ic('\n ** AUC (via metrics.auc): {}'.format(auc))
    ic(len(fpr), len(tpr), auc, len(threshold_v))

    return [fpr, tpr, auc, threshold_v]

def shuffler(is_lens):
    ic(' -- getting list of indexes to randomize data')
    liso = np.linspace(0, len(is_lens)-1, (len(is_lens)))
    np.random.shuffle(liso)
    liso = liso.astype(int)
    #indexes = np.array(indexes)
    return liso

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