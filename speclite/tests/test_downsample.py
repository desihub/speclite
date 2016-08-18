# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..downsample import downsample
import numpy as np
import numpy.ma as ma


def test_identity_unweighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_in[2] = (-2., 3.)
    data_out = downsample(data_in, 1)
    assert np.all(np.asarray(data_out) == np.asarray(data_in))


def test_identity_weighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_in[2] = (-2., 3.)
    data_out = downsample(data_in, 1, weight='y')
    assert np.all(np.array(data_out) == np.array(data_in))


def test_constant_unweighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2)
    assert np.array_equal(data_out, data_in[:5])


def test_constant_weighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2, weight='y')
    assert np.all(data_out['x'] == 1.)
    assert np.all(data_out['y'] == 2.)


def test_masked_unweighted():
    data_in = ma.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2)
    assert ma.isMA(data_out)
    assert np.array_equal(data_out, data_in[:5])
    data_in['x'].mask[2] = True
    data_in.mask[7] = (True, True)
    data_out = downsample(data_in, 2)
    assert np.array_equal(data_out, data_in[:5])


def test_masked_weighted():
    data_in = ma.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2, weight='y')
    assert ma.isMA(data_out)
    assert np.all(data_out['x'] == 1.)
    assert np.all(data_out['y'] == 2.)
    data_in['x'].mask[2] = True
    data_in.mask[7] = (True, True)
    data_out = downsample(data_in, 2, weight='y')
    assert np.all(data_out['x'] == 1.)
    assert np.all(data_out['y'] == (2., 1., 2., 1., 2.))


def test_masked_weighted_invalid():
    data_in = ma.ones((10,), dtype=[('x', float), ('y', float)])
    data_in['x'][2] = np.inf
    data_in['x'].mask[2] = True
    data_in['y'][7] = np.nan
    data_in['y'].mask[7] = True
    data_out = downsample(data_in, 2, weight='y')
    assert np.all(data_out['x'] == 1.)
    assert np.all(data_out['y'] == (2., 1., 2., 1., 2.))


def test_no_trim():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 3, auto_trim=False)


def test_auto_trim():
    data_in = np.ones((4,), dtype=[('x', float), ('y', float)])
    data_in['x'] = np.arange(4)
    data_out = downsample(data_in, 2, start_index=0)
    assert np.all(data_out['x'] == (0.5, 2.5))
    data_out = downsample(data_in, 2, start_index=1)
    assert np.all(data_out['x'] == (1.5, ))
    data_out = downsample(data_in, 2, start_index=2)
    assert np.all(data_out['x'] == (2.5, ))


def test_invalid_data_in():
    with pytest.raises(ValueError):
        downsample('invalid', 1)


def test_invalid_axis():
    data_in = np.ones((1, 2, 3,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 1, axis=4)
    with pytest.raises(ValueError):
        downsample(data_in, 1, axis=-4)


def test_invalid_data_out():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 1, data_out='invalid')


def test_invalid_args():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 0)
    with pytest.raises(ValueError):
        downsample(data_in, 11)
    with pytest.raises(ValueError):
        downsample(data_in, 1, start_index=-1)
    with pytest.raises(ValueError):
        downsample(data_in, 1, start_index=10)
    with pytest.raises(ValueError):
        downsample(data_in, 5, start_index=6)


def test_no_trim():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 3, auto_trim=False)


def test_invalid_weight():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        downsample(data_in, 1, weight='z')
    with pytest.raises(ValueError):
        downsample(data_in, 1, weight=123)
    with pytest.raises(ValueError):
        data_in['y'][1] = -1
        downsample(data_in, 1, weight='y')


def test_bad_data_out_shape():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        data_out = np.copy(data_in)
        downsample(data_in, 2, data_out=data_out)
    with pytest.raises(ValueError):
        data_out = np.copy(data_in[:9])
        downsample(data_in, 1, data_out=data_out)


def test_bad_data_out_dtype():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    with pytest.raises(ValueError):
        data_out = np.ones((10,), dtype=[('x', float), ('z', float)])
        downsample(data_in, 1, data_out=data_out)
    with pytest.raises(ValueError):
        data_out = np.ones((10,), dtype=[('x', float), ('y', int)])
        downsample(data_in, 1, data_out=data_out)
    with pytest.raises(ValueError):
        data_out = np.ones((10,), dtype=[('x', float),])
        downsample(data_in, 1, data_out=data_out)
