import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skimage import transform
from skimage._shared.utils import _supported_float_type


def test_dct_matrix_3by3():
    matrix = transform.dct_matrix(3)
    sq2 = math.sqrt(2)
    sq3 = math.sqrt(3)
    sq6 = math.sqrt(6)
    expected = np.asarray([[1/sq3, 1/sq3, 1/sq3],
                           [1/sq2, 0, -1/sq2],
                           [1/sq6, -2/sq6, 1/sq6]])
    assert_allclose(matrix, expected, atol=1e-12)


@pytest.mark.parametrize('matrix', ['dct', 'klt'])
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.uint8])
def test_transform_channels(matrix, channel_axis, dtype):
    rng = np.random.default_rng(1234)
    shape = (16, 8, 3)
    if np.dtype(dtype).kind == 'f':
        x = rng.standard_normal(shape)
    else:
        x = rng.integers(0, 255, shape)
    x = x.astype(dtype, copy=False)
    x = np.moveaxis(x, source=-1, destination=channel_axis)

    # forward transform
    y, matrix = transform.transform_channels(x, matrix,
                                             channel_axis=channel_axis)
    assert y.shape == x.shape
    float_type = _supported_float_type(x.dtype)
    assert y.dtype == float_type
    # orthonormal transform preserves the norm
    x_norm = np.linalg.norm(x.astype(float, copy=False))
    y_norm = np.linalg.norm(y.astype(float, copy=False))
    assert abs(x_norm - y_norm) < 1e-5

    # complete the round trip transform
    r, _ = transform.transform_channels(y, matrix, inverse=True,
                                        channel_axis=channel_axis)
    assert r.shape == x.shape
    assert r.dtype == _supported_float_type(x.dtype)
    if float_type == np.float64:
        rtol = atol = 1e-12
    else:
        rtol = atol = 1e-6
    assert_allclose(x, r, rtol=rtol, atol=atol)
