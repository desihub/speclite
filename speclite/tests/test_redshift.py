# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..redshift import _redshift, transform, apply_redshift_transform
import numpy as np
import numpy.ma as ma
import astropy.table
import astropy.units as u


def test_z_par():
    # Must either provide z or both z_in and z_out.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'flux'))
    with pytest.raises(ValueError):
        transform(t)
    with pytest.raises(ValueError):
        transform(t, z=1, z_out=0)
    with pytest.raises(ValueError):
        transform(t, z=1, z_in=0)
    with pytest.raises(ValueError):
        transform(t, z_in=0)
    with pytest.raises(ValueError):
        transform(t, z_out=0)
    # Values must be > -1.
    with pytest.raises(ValueError):
        transform(t, z=-1)
    with pytest.raises(ValueError):
        transform(t, z_in=-1, z_out=0)
    with pytest.raises(ValueError):
        transform(t, z_in=0, z_out=-1)
    # Passing z is equivalent to passing z_out=z and z_in=0.
    t2 = transform(t, z=1.)
    t3 = transform(t, z_in=0., z_out=1.)
    assert np.all(t2['wlen'] == t3['wlen'])
    assert np.all(t2['flux'] == t3['flux'])


def test_table():
    # Exponents are inferred from column names.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'flux'))
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Transforms can be applied in place.
    tc = transform(t, z=1., in_place=True)
    assert tc is t
    assert np.all(t['wlen'] == t2['wlen'])
    assert np.all(t['flux'] == t2['flux'])
    # Unrecognized names are passed through.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'extra'))
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['extra'] == t['extra'])
    # Any input units are preserved.
    t = astropy.table.Table()
    t['wlen'] = np.array([1., 2., 3.]) * u.Angstrom
    t['flux'] = np.ones(3) * u.erg / (u.s * u.cm ** 2 * u.Angstrom)
    t2 = transform(t, z=1.)
    assert t2['wlen'].unit == t['wlen'].unit
    assert t2['flux'].unit == t['flux'].unit
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Any input masks are preserved.
    t = astropy.table.Table([[1., 2., 3.], [1., 1., 1.]],
                            names=('wlen', 'flux'), masked=True)
    t['flux'].mask[1] = True
    t2 = transform(t, z=1.)
    assert np.all(t['flux'].mask == t2['flux'].mask)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])


def test_structured_array():
    # Exponents are inferred from column names.
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Transforms can be applied in place.
    tc = transform(t, z=1., in_place=True)
    assert tc is t
    assert np.all(t['wlen'] == t2['wlen'])
    assert np.all(t['flux'] == t2['flux'])
    # Unrecognized names are passed through.
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('extra', float)])
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['extra'] == t['extra'])
    # Any input units are preserved.
    t = np.array([(1,), (2,), (3,)], dtype=[('wlen', float)]) * u.Angstrom
    t2 = transform(t, z=1.)
    assert t2['wlen'].unit == t['wlen'].unit
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    # Any input masks are preserved.
    t = ma.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    t['flux'].mask[1] = True
    t2 = transform(t, z=1.)
    assert np.all(t['flux'].mask == t2['flux'].mask)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])


def test_passthru():
    data_in = np.zeros((4, 5))
    result = _redshift(z_in=0, z_out=0, data_in=data_in)
    assert np.array_equal(data_in, result)


def test_invalid_z():
    with pytest.raises(ValueError):
        _redshift(z_in='invalid', z_out=0)
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out='invalid')


def test_negative_z():
    with pytest.raises(ValueError):
        _redshift(z_in=-1, z_out=0)
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=-1)
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=np.arange(-1, 1))


def test_incompatible_z():
    with pytest.raises(ValueError):
        _redshift(z_in=np.arange(2), z_out=np.arange(3))


def test_broadcast_shape():
    z_in = np.zeros((2, 1, 1, 1))
    z_out = np.zeros((1, 3, 1, 1))
    data_in = np.zeros((4, 5))
    result = _redshift(z_in=z_in, z_out=z_out, data_in=data_in)
    assert result.shape == (2, 3, 4, 5), 'Invalid broadcast shape.'


def test_bad_data_type():
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in='invalid')
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_out='invalid')


def test_invalid_rule_value():
    data_in = np.zeros((5,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in,
                  rules=[{'name': 123, 'exponent': 1}])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in,
                  rules=[{'name': 'wlen', 'exponent': 'invalid'}])
    with pytest.raises(ValueError):
        wlen = np.zeros(5, float)
        _redshift(z_in=0, z_out=0, data_in=data_in,
                  rules=[{'name': 'wlen', 'exponent': 1, 'array_in': wlen}])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0,
                  rules=[{'name': 'wlen', 'exponent': 1, 'array_in': 0}])


def test_incomplete_rule():
    data_in = np.zeros((4, 5))
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, rules=[{'name': 'wlen'}])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, rules=[{'exponent': 0}])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, rules=[{'name': 'wlen', 'exponent': 0}])


def test_conflicting_rule():
    data_in = np.zeros((5,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, rules=[
            {'name': 'wlen', 'exponent': 0, 'array_in': np.arange(5)}])


def test_missing_name():
    data_in = np.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in,
                  rules=[{'name': 'ivar', 'exponent': 2}])
    data_out = np.zeros((10,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': 1}, {'name': 'flux', 'exponent': -1}])


def test_incompatible_array_shapes():
    wlen = np.arange(10)
    flux = np.arange(11)
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
            {'name': 'flux', 'exponent': -1, 'array_in': flux}])


def test_incompatible_array_out_shapes():
    wlen = np.zeros(10, dtype=np.float32)
    data_out = np.empty(11, dtype=[('wlen', np.float32)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen}])


def test_incompatible_array_out_types():
    wlen = np.zeros(10, dtype=np.float32)
    data_out = np.empty(10, dtype=[('wlen', np.float64)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen}])


def test_incompatible_in_out_shapes():
    data_in = np.empty(10, dtype=[('wlen', np.float32)])
    data_out = np.empty(11, dtype=[('wlen', np.float32)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1}])


def test_incompatible_in_out_types():
    data_in = np.empty(10, dtype=[('wlen', np.float32)])
    data_out = np.empty(10, dtype=[('wlen', np.float64)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1}])


def test_discarded_mask():
    data_in = ma.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    data_out = np.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_in=data_in, data_out=data_out)
    wlen = np.arange(10)
    flux = ma.zeros((10,))
    with pytest.raises(ValueError):
        _redshift(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
            {'name': 'flux', 'exponent': -1, 'array_in': flux}])


def test_array_in_round_trip():
    wlen = np.arange(10)
    flux = np.ones((10,))
    result1 = _redshift(z_in=0, z_out=1, rules=[
        {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
        {'name': 'flux', 'exponent': -1, 'array_in': flux}])
    assert result1.dtype.names == ('wlen', 'flux'), 'Invalid output names.'
    assert result1.shape == (10,), 'Invalid result1 shape.'
    result2 = _redshift(z_in=1, z_out=0, data_in=result1, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result2.dtype.names == ('wlen', 'flux'), 'Invalid output names.'
    assert result2.shape == (10,), 'Invalid result2 shape.'
    assert np.allclose(result2['wlen'], wlen)
    assert np.allclose(result2['flux'], flux)


def test_data_in_round_trip():
    data_in = np.empty((10,), dtype=[
        ('wlen', float), ('flux', float), ('extra', int)])
    data_in['wlen'] = np.arange(10)
    data_in['flux'] = np.ones((10,))
    data_in['extra'] = np.arange(10)
    result1 = _redshift(z_in=0, z_out=1, data_in=data_in, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result1.dtype.names == ('wlen', 'flux', 'extra'),\
        'Invalid output names.'
    assert result1.shape == (10,), 'Invalid result1 shape.'
    result2 = _redshift(z_in=1, z_out=0, data_in=result1, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result2.dtype.names == ('wlen', 'flux', 'extra'),\
        'Invalid output names.'
    assert result2.shape == (10,), 'Invalid result2 shape.'
    assert np.array_equal(result2['wlen'], data_in['wlen'])
    assert np.array_equal(result2['flux'], data_in['flux'])
    assert np.array_equal(result2['extra'], data_in['extra'])


def test_propagated_array_mask():
    wlen = np.arange(10)
    flux = ma.ones((10,))
    flux.mask = False
    flux[2] = ma.masked
    result = _redshift(z_in=0, z_out=1, rules=[
        {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
        {'name': 'flux', 'exponent': -1, 'array_in': flux}])
    assert ma.isMA(result)
    assert not result['wlen'].mask[2], 'Input mask not propagated.'
    assert result['flux'].mask[2], 'Input mask not propagated.'
    assert not result['flux'].mask[3], 'Input mask not propagated.'


def test_propagated_data_mask():
    data_in = ma.ones((10,), dtype=[
        ('wlen', float), ('flux', float), ('extra', int)])
    data_in['wlen'][1] = ma.masked
    data_in['extra'][2] = ma.masked
    result = _redshift(z_in=0, z_out=1, data_in=data_in, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert ma.isMA(result)
    assert not result['wlen'].mask[0], 'Input mask not propagated.'
    assert not result['flux'].mask[0], 'Input mask not propagated.'
    assert result['wlen'].mask[1], 'Input mask not propagated.'
    assert result['extra'].mask[2], 'Input mask not propagated.'
