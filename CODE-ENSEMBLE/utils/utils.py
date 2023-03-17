import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD
from keras.optimizers import Adam
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
import model_lib
from icecream import ic
import sklearn

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

def ic_images_clecio_like(x_data, num_ics, num_channels, input_shape):
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

    for sm in range(num_ics):
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
        plt.savefig('img_I_{}.png'.format(sm))

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
        plt.savefig('img_I_D_{}.png'.format(sm))

        for bu in range(num_ics*100):
            if os.path.exists(source+'./img_I_D_{}.png'.format(sm)):
                shutil.move(source+'./img_I_D_{}.png'.format(sm), dest1)
                counter = counter + 1
            if os.path.exists(source+'./img_I_{}.png'.format(sm)):
                shutil.move(source+'./img_I_{}.png'.format(sm), dest1)
                counter = counter + 1

    ic("\n ** Done. %s files moved." % counter)

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

def ic_images(x_data, num_samples):
    count = 0
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')
    PATH = os.getcwd()

    for xe in range(0,num_samples,1):
        x_line = x_data[xe,:,:,:]
        ic("x_line shape:", x_line.shape)
        #x_line = x_data[:][:][:][xe]
        rgb = toimage(x_line)
        rgb = np.array(rgb)
        im1 = Image.fromarray(rgb)
        #im1 = im1.resize((101,101), Image.ANTIALIAS)
        #cv2.resize(im1, (84, 84))
        im1.save("img_Y_%s.png" % xe)
        count = count + 1

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

    for bu in range(0, num_samples, 1):
        if os.path.exists(source+'./img_Y_%s.png' % bu):
            shutil.move(source+'./img_Y_%s.png' % bu, dest1)
            counter = counter + 1

    ic("\n ** Done. %s files moved." % counter)

def save_clue(x_data, y_data, TR, version, step, input_shape, nrows, ncols, index):
    figcount = 0
    plt.figure()
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20))
    for i in range(nrows):
        for j in range(ncols):
            temp_image = toimage(np.array(x_data[figcount, :, :, :]))
            axs[i, j].imshow(temp_image)
            axs[i, j].set_title('Class: %s' % y_data[figcount])
            figcount = figcount + 1

    index = index + 1
    plt.savefig("CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png". format(TR, version, step, input_shape, input_shape, index))
    return figcount

# Separa os conjuntos de treino, teste e validação.
def test_samples_balancer(y_data, x_data, vallim, train_size, percent_data, challenge):
    y_size = len(y_data)
    y_yes, y_no, y_excess = ([] for i in range(3))
    e_lente = 0
    n_lente = 0
    if challenge == 'challenge1':
        if train_size > 1600:
            ic(' ** Using Temporary train_size')
            train_size = train_size/10
        else:
            ic(' ** Using Regular train_size')
    # range(start, stop, step) - (início do intervalo, fim, o quanto aumenta em cada loop)
    for y in range(0, y_size, 1):
        if y_data[y] == 1:
            # Pegamos uma quantidade de dados para treino e o que sobra vai para o excess e é usado para validação/teste
            e_lente += 1
            if len(y_yes) < (train_size * 5):
                # Armazenamos os índices
                y_yes = np.append(int(y), y_yes)
            else:
                y_excess = np.append(int(y), y_excess)
        else:
            n_lente += 1
            if len(y_no) < (train_size * 5):
                y_no = np.append(int(y), y_no)
            else:
                y_excess = np.append(int(y), y_excess)

    ic(' Casos lente = ', e_lente)
    ic(' Casos nao lente = ', n_lente)
    y_y = np.append(y_no, y_yes)
    np.random.shuffle(y_y)

    np.random.shuffle(y_excess)
    y_y = y_y.astype(int)
    y_excess = np.array(y_excess)
    y_excess = y_excess.astype(int)

    # Define o tamanho do conjunto de validação, utilizando a variável vallim (nesse caso 2.000)
    y_val = y_data[y_excess[0:vallim]]
    x_val = x_data[y_excess[0:vallim]]

    y_test = y_data[y_excess[vallim:int(len(y_excess)*percent_data)]]
    x_test = x_data[y_excess[vallim:int(len(y_excess)*percent_data)]]
    ic('Realizando testes com conjunto normal de dados. Dados de teste = ', len(x_test))

    # Preenchemos o y_data, usando os índices criados no y_y
    y_data = y_data[y_y]
    x_data = x_data[y_y]

    return [y_data, x_data, y_test, x_test, y_val, x_val]


# Randomiza os dados para divisão nas folds
def load_data_kfold(k, x_data, y_data):
    ic('Preparing Folds')
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_data, y_data))

    return folds


def get_callbacks(name_weights, patience_lr, name_csv):
    mcp_save = ModelCheckpoint(name_weights, save_best_only = True, monitor = 'val_loss', mode = 'min')
    csv_logger = CSVLogger(name_csv)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4,
                                       mode='max')
    return [mcp_save, csv_logger, reduce_lr_loss]


def select_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        ic('\n ** Usando otimizador: ', optimizer)
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    else:
        ic('\n ** Usando otimizador: ', optimizer)
        opt = Adam(learning_rate=learning_rate)
    return opt


# Gera a curva ROC
def roc_curve_calculate(y_test, x_test, model, rede):
    ic('Roc Curve Calculating')
    if rede == 'ensemble':
        ic('\n ** Preds: ', rede)
        resnet = model[0]
        efn = model[1]

        prob_resnet = resnet.predict(x_test)
        prob_efn = efn.predict(x_test)

        ic('\n ** prob_resnet: ', prob_resnet)
        ic('\n ** prob_efn: ', prob_efn)

        probs = (prob_resnet + prob_efn) / 2
        ic('\n ** probs_ensemble: ', probs)

    else:
        ic('\n ** Preds: ', rede)
        probs = model.predict(x_test)
        ic('\n ** probs: ', probs)
        ic('\n ** probs.shape: ', probs.shape)

    probsp = probs[:, 1]
    # ic('\n ** probsp: ', probsp)
    # ic('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    ic('\n ** y_new: ', y_new)
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
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

    auc2 = roc_auc_score(y_test[:, 1], probsp)
    auc = metrics.auc(fpr, tpr)

    ic('\n ** AUC (via metrics.auc): {}, AUC (via roc_auc_score): {}'.format(auc, auc2))
    # ic('\n ** TP_Rate: {}'.format(tpr))
    # ic('\n ** FP_Rate: {}'.format(fpr))
    # ic('\n ** AUC: {}'.format(auc))
    # ic('\n ** AUC2: {}'.format(auc2))
    # ic('\n ** Thresh: {}'.format(thres))

    return [tpr, fpr, auc, auc2, thres]

def roc_curve_calculate_ensemble(y_test, x_test, model_resnet, model_effnet, rede, version):
    ic('Roc Curve Calculating')
    ic('\n ** Preds: ', rede)
    # resnet = model[0]
    # efn = model[1]

    prob_resnet = model_resnet.predict(x_test)
    prob_efn = model_effnet.predict(x_test)
    ic(" ** There goes the Probabilities vector:")
    ic(" ** Resnet:")
    ic(prob_resnet)
    ic(" ** EfficientNet:")
    ic(prob_efn)

    with open('ensemble_data_%s.csv' % version, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[prob_resnet[:,1]], [prob_efn[:,1]]])
        #writer.writerows(code_data)

    # ic('\n ** prob_resnet: ', prob_resnet)
    # ic('\n ** prob_efn: ', prob_efn)

    probs = (prob_resnet + prob_efn) / 2
    ic('\n ** probs_ensemble: ', probs)

    probsp = probs[:, 1]
    # ic('\n ** probsp: ', probsp)
    # ic('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    # ic('\n ** y_new: ', y_new)
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
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

    auc2 = roc_auc_score(y_test[:, 1], probsp)
    auc = metrics.auc(fpr, tpr)

    ic('\n ** AUC (via metrics.auc): {}, AUC (via roc_auc_score): {}'.format(auc, auc2))
    # ic('\n ** TP_Rate: {}'.format(tpr))
    # ic('\n ** FP_Rate: {}'.format(fpr))
    # ic('\n ** AUC: {}'.format(auc))
    # ic('\n ** AUC2: {}'.format(auc2))
    # ic('\n ** Thresh: {}'.format(thres))

    return [tpr, fpr, auc, auc2, thres]

def roc_curves_sec(y_test, x_test, models, model_list, version):
    ic('Roc Curve Calculating')
    ic('\n ** Models: ', model_list)
    # resnet = model[0]
    # efn = model[1]
    probas = []

    for i in range(len(model_list)):
        if i == 0:
            probas = models[i].predict(x_test)
        else:
            probas = ((models[i].predict(x_test))+probas)
    probs = probas / len(model_list)
    #print('\n ** probs_ensemble: ', probs)

    probsp = probs[:, 1]
    # ic('\n ** probsp: ', probsp)
    # ic('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    # ic('\n ** y_new: ', y_new)
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
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

    auc2 = roc_auc_score(y_test[:, 1], probsp)
    auc = metrics.auc(fpr, tpr)

    ic('\n ** AUC (via metrics.auc): {}, AUC (via roc_auc_score): {}'.format(auc, auc2))
    # ic('\n ** TP_Rate: {}'.format(tpr))
    # ic('\n ** FP_Rate: {}'.format(fpr))
    # ic('\n ** AUC: {}'.format(auc))
    # ic('\n ** AUC2: {}'.format(auc2))
    # ic('\n ** Thresh: {}'.format(thres))

    return [tpr, fpr, auc, auc2, thres]

def acc_score(acc0, history, val_acc0, loss0, val_loss0):
    ic('\n ** Calculating acc0, val_acc0, loss0, val_loss0')
    acc0 = np.append(acc0, history.history['accuracy'])
    val_acc0 = np.append(val_acc0, history.history['val_accuracy'])
    loss0 = np.append(loss0, history.history['loss'])
    val_loss0 = np.append(val_loss0, history.history['val_loss'])
    ic('\n ** Finished Calculating!')

    return [acc0, val_acc0, loss0, val_loss0]


def acc_score_ensemble(acc0, history, val_acc0, loss0, val_loss0):
    ic('\n ** Calculating acc0, val_acc0, loss0, val_loss0')
    acc0 = np.append(acc0, history.history['accuracy'])
    val_acc0 = np.append(val_acc0, history.history['val_accuracy'])
    loss0 = np.append(loss0, history.history['loss'])
    val_loss0 = np.append(val_loss0, history.history['val_loss'])
    ic('\n ** Finished Calculating!')

    return [acc0, val_acc0, loss0, val_loss0]

def FScore_curves(rede, model, x_test, y_test):
    ic(' ** F Scores Curve Calculating')
    ic('\n ** Preds: ', rede)
    probs = model.predict(x_test)
    ic('\n ** probs: ', probs)
    ic('\n ** probs.shape: ', probs.shape)

    probsp = probs[:, 1]
    # ic('\n ** probsp: ', probsp)
    # ic('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    ic('\n ** y_new: ', y_new)
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    prec, rec = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
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

        try:
            prec.append(tp_score/(tp_score+fp_score))
            rec.append(tp_score/(tp_score+fn_score))
        except:
            prec.append(0.0)
            rec.append(0.0)
    try:
        ic(len(y_test), len(probsp))
    except:
        ic(y_test.shape, probsp.shape)
    y_tests = np.argmax(y_test, axis = 1)
    f_1_score = sklearn.metrics.f1_score(y_tests, probsp)
    f_001_score = sklearn.metrics.fbeta_score(y_tests, probsp, beta=0.01)

    ic('\n ** F1 Score: {}, Fbeta Score: {}'.format(f_1_score, f_001_score))

    return [prec, rec, f_1_score, f_001_score, thres]

def FScore_curves_ensemble(y_test, x_test, models, model_list):
    ic(' ** FScore Curves Ensemble')
    ic(' ** Preds: ', model_list)
    probas = np.zeros((len(model_list)))

    for i in range(len(model_list)):
        probas[i] = models[i].predict(x_test)

    probs = (np.sum(probas[j] for j in len(model_list))) / len(model_list)
    ic('\n ** probs_ensemble: ', probs)

    probsp = probs[:, 1]
    # ic('\n ** probsp: ', probsp)
    # ic('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    ic('\n ** y_new: ', y_new)
    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    prec, rec = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
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
                    
        prec.append(tp_score/(tp_score+fp_score))
        rec.append(tp_score/(tp_score+fn_score))

    f_1_score = sklearn.metrics.f1_score(y_test, probsp)
    f_001_score = sklearn.metrics.fbeta_score(y_test, probsp, beta=0.01)

    ic('\n ** F1 Score: {}, Fbeta Score: {}'.format(f_1_score, f_001_score))

    return [prec, rec, f_1_score, f_001_score, thres]

def get_model_roulette(mod, x_data, weights):
    if mod == 'resnet50':
        resnet_depth = 50
        models = model_lib.get_model_resnet(x_data, weights, resnet_depth)
    if mod == 'resnet101':
        resnet_depth = 101
        models = model_lib.get_model_resnet(x_data, weights, resnet_depth)
    if mod == 'resnet152':
        resnet_depth = 152
        models = model_lib.get_model_resnet(x_data, weights, resnet_depth)
    if mod == 'effnet_B0':
        effnet_version = 'B0'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B1':
        effnet_version = 'B1'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B2':
        effnet_version = 'B2'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B3':
        effnet_version = 'B3'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B4':
        effnet_version = 'B4'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B5':
        effnet_version = 'B5'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B6':
        effnet_version = 'B6'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'effnet_B7':
        effnet_version = 'B7'
        models = model_lib.get_model_effnet(x_data, weights, effnet_version)
    if mod == 'inceptionV2':
        version = 'V2'
        models = model_lib.get_model_inception(x_data, weights, version)
    if mod == 'inceptionV3':
        version = 'V3'
        models = model_lib.get_model_inception(x_data, weights, version)
    if mod == 'xception':
        models = model_lib.get_model_xception(x_data, weights)

    return models

def create_path(study, inside=False, train_size=None):
    root_dir = '/home/kayque/LENSLOAD/SCRIPTS/TESTS/CODE-PACK-02/RESULTS/'
    if os.path.exists(root_dir):
        ic(' Results dir already exists')
    else:
        ic(' creating Results dir')
        os.makedirs(output_directory)
    output_directory = root_dir+study
    if os.path.exists(output_directory):
        ic(' %s dir already exists' % study)
    else:
        ic(' creating %s dir' % study)
        os.makedirs(output_directory)
    if inside == True:
        output_directory = output_directory+'%s/' % (train_size)
        if os.path.exists(output_directory): 
            return None
        else: 
            os.makedirs(output_directory)
