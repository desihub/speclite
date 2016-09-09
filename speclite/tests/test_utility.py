# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..utility import *
import numpy as np
import numpy.ma as ma


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


def test_tabular_round_trip():
    for mode in ('read_only', 'in_place'):
        # Numpy structured array.
        t = np.ones(3, dtype=[('x', float), ('y', float)])
        c, r = prepare_data(mode, [t], {})
        assert np.all(t == r)
        t2 = tabular_like(t, c)
        assert np.all(t == t2)
        # Numpy structured masked array.
        t = ma.ones(3, dtype=[('x', float), ('y', float)])
        c, r = prepare_data(mode, [t], {})
        assert np.all(t == r) and ma.isMaskedArray(r)
        t2 = tabular_like(t, c)
        assert np.all(t == t2) and ma.isMaskedArray(t2)
        # Astropy table
        t = astropy.table.Table([[1., 2., 3.], [1., 1., 1.]], meta=dict(a=1),
                                names=('x', 'y'))
        c, r = prepare_data(mode, [t], {})
        assert np.all(t == r) and t.meta == r.meta
        t2 = tabular_like(t, c)
        assert np.all(t == t2) and t.meta == t2.meta
        # Astropy masked table.
        t = astropy.table.Table([[1., 2., 3.], [1., 1., 1.]], meta=dict(a=1),
                                names=('x', 'y'), masked=True)
        c, r = prepare_data(mode, [t], {})
        assert np.all(t == r) and t.meta == r.meta and r.masked
        t2 = tabular_like(t, c)
        assert np.all(t == t2) and t.meta == t2.meta and t2.masked
        # Dictionary of numpy arrays.
        t = dict(x=np.arange(3), y=np.ones(3))
        # Pass dictionary as single positional arg.
        c, r = prepare_data(mode, [t], {})
        assert np.all(t['x'] == r['x']) and np.all(t['y'] == r['y'])
        t2 = tabular_like(t, c)
        assert np.all(t['x'] == t2['x']) and np.all(t['y'] == t2['y'])
        # Pass dictionary as individual keyword args.
        c, r = prepare_data(mode, [], t)
        assert np.all(t['x'] == r['x']) and np.all(t['y'] == r['y'])
        t2 = tabular_like(t, c)
        assert np.all(t['x'] == t2['x']) and np.all(t['y'] == t2['y'])
        # Dictionary of python lists.
        t = dict(x=[1., 2., 3.], y=[1., 1., 1.])
        if mode == 'read_only':
            c, r = prepare_data(mode, [t], {})
            assert np.all(t['x'] == r['x']) and np.all(t['y'] == r['y'])
            t2 = tabular_like(t, c)
            assert np.all(t['x'] == t2['x']) and np.all(t['y'] == t2['y'])
        else:
            # Cannot modify python lists in place.
            with pytest.raises(ValueError):
                c, r = prepare_data(mode, [t], {})
        # Ordered dictionary of python lists.
        t = collections.OrderedDict()
        t['y'] = [1., 1., 1.]
        t['x'] = [1., 2., 3.]
        if mode == 'read_only':
            c, r = prepare_data(mode, [t], {})
            assert np.all(t['x'] == r['x']) and np.all(t['y'] == r['y'])
            t2 = tabular_like(t, c)
            assert np.all(t['x'] == t2['x']) and np.all(t['y'] == t2['y'])
        else:
            # Cannot modify python lists in place.
            with pytest.raises(ValueError):
                c, r = prepare_data(mode, [t], {})


def test_tabular_dimension():
    t = np.ones((3, 1, 2), dtype=[('x', float), ('y', float)])
    c, r = prepare_data('read_only', [t], {})
    t2 = tabular_like(t, c, dimension=1)
    assert t2.shape == (3,) and t2['x'].shape == (3, 1, 2)
    t2 = tabular_like(t, c, dimension=2)
    assert t2.shape == (3, 1,) and t2['x'].shape == (3, 1, 2)
    t2 = tabular_like(t, c, dimension=3)
    assert t2.shape == (3, 1, 2,) and t2['x'].shape == (3, 1, 2)
