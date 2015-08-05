# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'][:] = np.arange(10., dtype=float)
    data['y'][:] = np.ones(10, dtype=float)
    x2 = np.arange(0.25, 9.25)
    result = resample(data, 'x', x2, 'y', kind='cubic')
    assert np.array_equal(result['x'], x2)
    assert np.allclose(result['y'], 1.)
    result = resample(data, 'x', x2, 'y', kind=3)
    assert np.array_equal(result['x'], x2)
    assert np.allclose(result['y'], 1.)
