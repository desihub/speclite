# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..resample import resample
import numpy as np
import numpy.ma as ma


def test_invalid_kind():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    x2 = np.arange(0.5, 9.5)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind='invalid')
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind=-1)


def test_one_y():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    x2 = np.arange(0.5, 9.5)
    result = resample(data, 'x', x2, 'y')
    assert result.shape == (9,), 'Unexpected result shape.'
    assert result.dtype == data.dtype, 'Unexpected result type.'
    assert np.array_equal(result['x'], x2)
    assert np.array_equal(result['y'], x2)


def test_two_ys():
    data = np.empty((10,), dtype=[('x', float), ('y1', float), ('y2', float)])
    data['x'] = np.arange(10.)
    data['y1'] = np.arange(10.)
    data['y2'] = 2 * np.arange(10.)
    x2 = x2 = np.arange(0.5, 9.5)
    result = resample(data, 'x', x2, ('y1', 'y2'))
    assert result.shape == (9,), 'Unexpected result shape.'
    assert result.dtype == data.dtype, 'Unexpected result type.'
    assert np.array_equal(result['x'], x2)
    assert np.array_equal(result['y1'], x2)
    assert np.array_equal(result['y2'], 2 * x2)


def test_extrapolate():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    x2 = np.arange(-1,11)
    result = resample(data, 'x', x2, 'y')
    assert ma.isMA(result)
    assert result['y'].mask[0]
    assert result['y'].mask[-1]
    assert np.array_equal(result['y'][1:-1], x2[1:-1])


def test_masked_all_valid():
    data = ma.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    data.mask = False
    x2 = np.arange(0.5, 9.5)
    result = resample(data, 'x', x2, 'y')
    assert np.array_equal(result['x'], x2)
    assert np.array_equal(result['y'], x2)


def test_masked_x():
    data = ma.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    data.mask = False
    data['x'].mask[2] = True
    x2 = np.arange(0.5, 9.5)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y')


def test_masked_one_invalid_linear():
    data = ma.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.arange(10.)
    data.mask = False
    data['y'].mask[4] = True
    x2 = np.arange(0.25, 9.25)
    result = resample(data, 'x', x2, 'y', kind='linear')
    assert np.array_equal(result['x'], x2)
    assert np.array_equal(result['y'][:3], x2[:3])
    assert result['y'].mask[3]
    assert result['y'].mask[4]
    assert np.array_equal(result['y'][5:], x2[5:])


def test_masked_one_invalid_nearest():
    data = ma.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.ones(10)
    data.mask = False
    data['y'].mask[4] = True
    x2 = np.arange(0.25, 9.25)
    result = resample(data, 'x', x2, 'y', kind='nearest')
    assert np.array_equal(result['x'], x2)
    assert np.all(result['y'][:4] == 1)
    assert result['y'].mask[4]
    assert np.all(result['y'][5:] == 1)


def test_masked_kind_not_supported():
    data = ma.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10.)
    data['y'] = np.ones(10)
    data.mask = False
    data['y'].mask[4] = True
    x2 = np.arange(0.25, 9.25)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind='quadratic')
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind='cubic')
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind=2)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind=3)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind=4)
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y', kind=5)


def test_cubic():
    # Cubic interpolation only works in numpy >= 1.8
    major, minor = tuple(np.__version__.split('.')[:2])
    if (int(major) == 1) and (int(minor) <= 7):
        return
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    x2 = np.arange(9) + 0.25
    result = resample(data, 'x', x2, 'y', kind=3)
    assert np.array_equal(result['x'], x2)
    assert np.allclose(result['y'], 1.)
    result = resample(data, 'x', x2, 'y', kind='cubic')
    assert np.array_equal(result['x'], x2)
    assert np.allclose(result['y'], 1.)

def test_data_in_invalid_type():
    #Invalid:  not a ndarray
    data_in = [0, 1, 2, 3, 4, 5]
    data_out = np.array([0, 1, 2, 3, 4, 5])
    x2 = np.arange(0.5, 4.5)
    with pytest.raises(ValueError):
        resample(data_in, data_out, x2, 'y')

    #Invalid:  ndarray, but not structured
    data = np.zeros((6,))
    with pytest.raises(ValueError):
        resample(data, data, x2, 'y')

    #Invalid:  ndarray, but multi-dim
    data = np.zeros((6,2), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        resample(data, 'x', x2, 'y')

def test_x_in_invalid_data():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)
    with pytest.raises(ValueError):
        resample(data, 'foobar', x2, 'y')

def test_x_in_invalid_type():
    #Invalid: x_in is not np.ndarray
    data = np.empty((10,), dtype=[('y', float)])
    data['y'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)
    x = [1]*10
    with pytest.raises(ValueError):
        resample(data, x, x2, 'y')

    #Invalid: x_in is ndarray, but dim does not match data
    x = np.ones((data.shape[0] + 1, ))
    with pytest.raises(ValueError):
        resample(data, x, x2, 'y')

    #Invalid:  x_in is masked AND actually has at least one masked value
    x = np.ma.array(np.ones(data.shape))
    x[0] = np.ma.masked
    with pytest.raises(ValueError):
        resample(data, x, x2, 'y')

def test_x_out_invalid_type():
    #Invlaid: x_out is not ndarray
    data = np.empty((10,), dtype=[('y', float)])
    data['y'][:] = np.ones(10, dtype=float)
    x = np.ones(data.shape)
    x2 = [1]*5
    with pytest.raises(ValueError):
        resample(data, x, x2, 'y')

def test_y_invalid_type():
    #Invalid: y type is not string and not iterable
    data = np.empty((10,), dtype=[('x', float), ('y', float), ('ytoo', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    data['ytoo'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)

    #It is actually hard to find something that makes sense that isn't also iterable;
    #so punting and just passing integer 12
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, 12)

    #Names with different types
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, ['y', 12])

    #y's with different types
    data = np.empty((10,), dtype=[('x', float), ('y', int), ('ytoo', float)])
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, ['y', 'ytoo'])

def test_y_invalid_data():
    data = np.empty((10,), dtype=[('x', float), ('y', float), ('ytoo', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    data['ytoo'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)

    #Non-existent names
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, 'foobar')

def test_data_out_invalid_type():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)

    #Invalid:  data_out has wrong shape
    data_out = np.empty((11,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, 'y', data_out=data_out)

    #Invalid: data_out has incorrect dtype
    data_out = np.empty((9,), dtype=[('x', int), ('y', int)])
    with pytest.raises(ValueError):
        result = resample(data, 'x', x2, 'y', data_out=data_out)
