# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..resample import resample
import numpy as np
import numpy.ma as ma


def test_single_y():
    data = np.empty((10,), dtype=[('x', float), ('y', float)])
    data['x'] = np.arange(10)
    data['y'] = np.arange(10)
    x2 = 0.5 + np.arange(9)
    result = resample(data, 'x', x2, 'y')
    assert result.shape == (9,), 'Unexpected result shape.'
    assert result.dtype == data.dtype, 'Unexpected result type.'
    assert np.array_equal(result['x'], x2)
    assert np.array_equal(result['y'], x2)
