"""Orthonormal transforms for channel energy compression."""
import math

import numpy as np

from skimage._shared.utils import _supported_float_type


def dct_matrix(n):
    """Compute an orthonormal DCT matrix for signals of size n.

    This matrix can be used to decorrelate channels in multichannel images.

    Parameters
    ----------
    n : int
        The number of samples (image channels) for the DCT matrix.

    Returns
    -------
    np.ndarray
        The shape ``(n, n)`` matrix corresponding to a 1D DCT Type II
        transform.

    Notes
    -----
    The output of this transform on an image's color channels has been referred
    to as the "opponent color space" [1]_. This transform can be advantageous
    for color image denoising (e.g. [2]_, [3]_).

    References
    ----------
    .. [1] K.N. Plataniotis and A.N. Venetsanopoulos. Color image processing
        and applications. Springer-Verlag New York, Inc., New York, NY, 2000.
        :DOI:10.1007/978-3-662-04186-4
    .. [2] A Foi, V Katkovnik, K Egiazarian. Pointwise Shape-Adaptive DCT for
        High-Quality Denoising and Deblocking of Grayscale and Color Images.
        IEEE Trans Image Processing 2007; 16(5): 1395-1411.
        :DOI:10.1109/TIP.2007.891788
    .. [3] G Yu and G Sapiro. DCT Image Denoising: a Simple and Effective Image
        Denoising Algorithm, Image Processing On Line, 1 (2011), pp. 292–296.
        :DOI:10.5201/ipol.2011.ys-dct
    """
    r = np.arange(n)
    sq_1_n = math.sqrt(1 / n)
    sq_2_n = math.sqrt(2 / n)
    pi_n = np.pi / n
    m = np.zeros((n, n))
    m[0, :] = sq_1_n
    for row in range(1, n):
        m[row, :] = sq_2_n * np.cos((r + 0.5) * row * pi_n)
    return m


def klt_matrix(x, channel_axis=-1):
    """Karhunen-Loève Transform (KLT) matrix along a channels dimension.

    Parameters
    ----------
    x : ndarray
        An n-dimensional array with multiple "channels" (e.g. R, G, B
        components for a color image) along ``channel_axis``.
    channel_axis : int, optional
        The axis along which to perform the KLT (i.e. PCA).

    Returns
    -------
    v : npdarray
        A matrix of shape (n_channels, n_channels) where
        ``n_channels = x.shape[channel_axis]``.

    Notes
    -----
    The KLT transform completely decorrelates the signal channels and
    compresses as much information as possible into as few channels as
    possible. In other words it is a principle component analysis (PCA) along
    the channels dimension.

    The KLT matrix, ``v = np.conj(u).T`` where u is a matrix containing the
    eigenvectors of the autocorrelation matrix of ``x`` as its vectors.

    ``y = np.dot(v, x)`` will has uncorrelated components.

    """
    if channel_axis < -x.ndim or channel_axis > (x.ndim - 1):
        raise ValueError("invalid axis")
    channel_axis = channel_axis % x.ndim
    nchannels = x.shape[channel_axis]
    if channel_axis != -1:
        x = x.swapaxes(-1, channel_axis)
    x = x.reshape((-1, nchannels), order="F")

    # compute the covariance matrix, x
    x = x - x.mean(axis=0, keepdims=1)
    if x.dtype.kind == "c":
        r = np.dot(np.conj(x).T, x)
    else:
        r = np.dot(x.T, x)

    # (symmetric) eigen decomposition
    _, v = np.linalg.eigh(r)

    # sort the eigenvectors in descending order instead
    v = v[:, ::-1]

    if x.dtype.kind == 'c':
        return np.conj(v).T
    else:
        return v.T


def transform_channels(x, matrix='klt', channel_axis=-1, inverse=False):
    if channel_axis < -x.ndim or channel_axis > (x.ndim - 1):
        raise ValueError("invalid axis")
    channel_axis = channel_axis % x.ndim
    n_channels = x.shape[channel_axis]
    float_dtype = _supported_float_type(x.dtype)
    x = x.astype(float_dtype, copy=False)

    if isinstance(matrix, str):
        # determine an orthonormal transform matrix
        if matrix == 'klt':
            matrix = klt_matrix(x, channel_axis=channel_axis)
        elif matrix == 'dct':
            matrix = dct_matrix(n_channels).astype(x.real.dtype)
        else:
            raise ValueError("unknown channel transform: {matrix}")
    elif isinstance(matrix, np.ndarray):
        if not (matrix.ndim == 2
                and all(s == n_channels for s in matrix.shape)):
            raise ValueError(
                "user provided channel_transform matrix must have shape "
                "(nchannels, nchannels)"
            )
    matrix = np.asarray(matrix, dtype=float_dtype)
    if inverse:
        # matrix is orthonormal, so its inverse is the (conjugate) transpose
        if matrix.dtype.kind == 'c':
            matrix = np.conj(matrix.T)
        else:
            matrix = matrix.T

    # apply the transform via matrix multiplication along the channels axis
    if channel_axis != 0:
        x = np.moveaxis(x, source=channel_axis, destination=0)
    orig_shape = x.shape
    y = np.dot(matrix, x.reshape((n_channels, -1), order="F"))
    y = y.reshape(orig_shape, order="F")
    if channel_axis != 0:
        y = np.moveaxis(y, source=0, destination=channel_axis)
    return y, matrix
