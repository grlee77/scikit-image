from skimage._shared.utils import (copy_func, assert_nD, abs_sq)
import numpy.testing as npt
import numpy as np
from skimage._shared import testing
import pytest


def test_assert_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        assert_nD(x, 2)


def test_copyfunc():
    def foo(a):
        return a

    bar = copy_func(foo, name='bar')
    other = copy_func(foo)

    npt.assert_equal(bar.__name__, 'bar')
    npt.assert_equal(other.__name__, 'foo')

    other.__name__ = 'other'

    npt.assert_equal(foo.__name__, 'foo')


pytest.mark.parametrize(
    'dtype, tol',
    [(np.float32, 1e-6),
     (np.float64, 1e-14),
     (np.complex64, 1e-6),
     (np.complex128, 1e-14)]
)
def test_abs_sq(dtype, tol):
    shape = (512, 512)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape).astype(dtype, copy=False)
    if x.dtype.kind == 'c':
        x += 1j * rstate.standard_normal(shape).astype(dtype, copy=False)

    npt.assert_allclose(np.abs(x)**2, abs_sq(x), atol=tol, rtol=tol)


if __name__ == "__main__":
    npt.run_module_suite()
