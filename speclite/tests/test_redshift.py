from astropy.tests.helper import pytest
from ..redshift import transform
import numpy as np
import numpy.ma as ma
import astropy.units as u


def test_passthru():
    data_in = np.zeros((4, 5))
    result = transform(z_in=0, z_out=0, data_in=data_in)
    assert np.array_equal(data_in, result)


def test_invalid_z():
    with pytest.raises(ValueError):
        transform(z_in='invalid', z_out=0)
    with pytest.raises(ValueError):
        transform(z_in=0, z_out='invalid')


def test_negative_z():
    with pytest.raises(ValueError):
        transform(z_in=-1, z_out=0)
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=-1)
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=np.arange(-1,1))


def test_incompatible_z():
    with pytest.raises(ValueError):
        transform(z_in=np.arange(2), z_out=np.arange(3))


def test_broadcast_shape():
    z_in = np.zeros((2, 1, 1, 1))
    z_out = np.zeros((1, 3, 1, 1))
    data_in = np.zeros((4, 5))
    result = transform(z_in=z_in, z_out=z_out, data_in=data_in)
    assert result.shape == (2, 3, 4, 5), 'Invalid broadcast shape.'


def test_bad_data_type():
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in='invalid')
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_out='invalid')


def test_invalid_rule_value():
    data_in = np.zeros((5,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, rules=[{'name': 123, 'exponent': 1}])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in,
            rules=[{'name': 'wlen', 'exponent': 'invalid'}])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in,
            rules=[{'name': 'wlen', 'exponent': 1, 'array_in': 0}])


def test_incomplete_rule():
    data_in = np.zeros((4, 5))
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, rules=[{'name': 'wlen'}])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, rules=[{'exponent': 0}])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, rules=[{'name': 'wlen', 'exponent': 0}])


def test_conflicting_rule():
    data_in = np.zeros((5,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in,
            rules=[{'name': 'wlen', 'exponent': 0, 'array_in': np.arange(5)}])


def test_missing_name():
    data_in = np.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, rules=[{'name': 'ivar', 'exponent': 2}])
    data_out = np.zeros((10,), dtype=[('wlen', float)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, data_out=data_out,
            rules=[{'name': 'wlen', 'exponent': 1}, {'name': 'flux', 'exponent': -1}])


def test_incompatible_array_shapes():
    wlen = np.arange(10)
    flux = np.arange(11)
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
            {'name': 'flux', 'exponent': -1, 'array_in': flux}])


def test_incompatible_array_out_shapes():
    wlen = np.zeros(10, dtype=np.float32)
    data_out = np.empty(11, dtype=[('wlen', np.float32)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen}])


def test_incompatible_array_out_types():
    wlen = np.zeros(10, dtype=np.float32)
    data_out = np.empty(10, dtype=[('wlen', np.float64)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen}])


def test_incompatible_in_out_shapes():
    data_in = np.empty(10, dtype=[('wlen', np.float32)])
    data_out = np.empty(11, dtype=[('wlen', np.float32)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1}])


def test_incompatible_in_out_types():
    data_in = np.empty(10, dtype=[('wlen', np.float32)])
    data_out = np.empty(10, dtype=[('wlen', np.float64)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1}])


def test_discarded_mask():
    data_in = ma.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    data_out = np.zeros((10,), dtype=[('wlen', float), ('flux', float)])
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_in=data_in, data_out=data_out)
    wlen = np.arange(10)
    flux = ma.zeros((10,))
    with pytest.raises(ValueError):
        transform(z_in=0, z_out=0, data_out=data_out, rules=[
            {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
            {'name': 'flux', 'exponent': -1, 'array_in': flux}])


def test_array_in_round_trip():
    wlen = np.arange(10)
    flux = np.ones((10,))
    result1 = transform(z_in=0, z_out=1, rules=[
        {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
        {'name': 'flux', 'exponent': -1, 'array_in': flux}])
    assert result1.dtype.names == ('wlen', 'flux'), 'Invalid output names.'
    assert result1.shape == (10,), 'Invalid result1 shape.'
    result2 = transform(z_in=1, z_out=0, data_in=result1, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result2.dtype.names == ('wlen', 'flux'), 'Invalid output names.'
    assert result2.shape == (10,), 'Invalid result2 shape.'
    assert np.allclose(result2['wlen'], wlen)
    assert np.allclose(result2['flux'], flux)


def test_data_in_round_trip():
    data_in = np.empty((10,), dtype=[('wlen', float), ('flux', float), ('extra', int)])
    data_in['wlen'] = np.arange(10)
    data_in['flux'] = np.ones((10,))
    data_in['extra'] = np.arange(10)
    result1 = transform(z_in=0, z_out=1, data_in=data_in, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result1.dtype.names == ('wlen', 'flux', 'extra'), 'Invalid output names.'
    assert result1.shape == (10,), 'Invalid result1 shape.'
    result2 = transform(z_in=1, z_out=0, data_in=result1, rules=[
        {'name': 'wlen', 'exponent': +1},
        {'name': 'flux', 'exponent': -1}])
    assert result2.dtype.names == ('wlen', 'flux', 'extra'), 'Invalid output names.'
    assert result2.shape == (10,), 'Invalid result2 shape.'
    assert np.array_equal(result2['wlen'], data_in['wlen'])
    assert np.array_equal(result2['flux'], data_in['flux'])
    assert np.array_equal(result2['extra'], data_in['extra'])


def test_propagated_array_mask():
    wlen = np.arange(10)
    flux = ma.ones((10,))
    flux.mask = False
    flux[2] = ma.masked
    result = transform(z_in=0, z_out=1, rules=[
        {'name': 'wlen', 'exponent': +1, 'array_in': wlen},
        {'name': 'flux', 'exponent': -1, 'array_in': flux}])
    assert ma.isMA(result)
    assert not result['wlen'].mask[2], 'Input mask not propagated.'
    assert result['flux'].mask[2], 'Input mask not propagated.'
    assert not result['flux'].mask[3], 'Input mask not propagated.'
