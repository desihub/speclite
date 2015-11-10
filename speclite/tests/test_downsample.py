# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..downsample import downsample
import numpy as np
import numpy.ma as ma


def test_identity_unweighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_in[2] = (-2., 3.)
    data_out = downsample(data_in, 1)
    assert np.array_equal(data_out, data_in)


def test_identity_weighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_in[2] = (-2., 3.)
    data_out = downsample(data_in, 1, weight='y')
    assert np.array_equal(data_out, data_in)


def test_constant_unweighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2)
    assert np.array_equal(data_out, data_in[:5])


def test_constant_weighted():
    data_in = np.ones((10,), dtype=[('x', float), ('y', float)])
    data_out = downsample(data_in, 2, weight='y')
    assert np.all(data_out['x'] == 1.)
    assert np.all(data_out['y'] == 2.)
