# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..utility import get_options, prepare_data
import numpy as np


def test_get_options_none():
    defaults = dict(x=1, y=2, z=3)
    kwin = dict(a=1, b=2, c=3)
    kwout, opt = get_options(kwin, **defaults)
    assert kwout == kwin
    assert opt == defaults


def test_get_options_some():
    defaults = dict(x=1, y=2, z=3)
    kwin = dict(a=1, b=2, c=3, x=-1)
    kwout, opt = get_options(kwin, **defaults)
    assert kwout == dict(a=1, b=2, c=3)
    assert opt == dict(x=-1, y=2, z=3)


def test_prepare_bad_mode():
    with pytest.raises(ValueError):
        prepare_data(mode='bad', args=[], kwargs={})


def test_prepare_bad_args():
    with pytest.raises(ValueError):
        prepare_data('read_only', args=['arg1', 'arg2'], kwargs={})
    with pytest.raises(ValueError):
        prepare_data('read_only', args=['arg'], kwargs=dict(kw='kw'))


def test_prepare_conv():
    # A single arg that is actually a dictionary is converted to kwargs.
    prepare_data('read_only', args=[dict(kw='kw')], kwargs={})


def test_prepare_bad_tabular():
    with pytest.raises(ValueError):
        prepare_data('read_only', args=['non-tabular'], kwargs={})


def test_prepare_bad_in_place():
    with pytest.raises(ValueError):
        prepare_data('in_place', args=[], kwargs={'x': [0, 1, 2]})


def test_prepare_bad_kwarg():
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    with pytest.raises(ValueError):
        prepare_data('read_only', args=[], kwargs={'t': t})
