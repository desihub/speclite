from astropy.tests.helper import pytest
from ..redshift import transform
import numpy as np


def test_negative_z():
    wavelength = np.arange(10)
    flux = np.zeros(shape=(10,))
    with pytest.raises(ValueError):
        transform(z=-1, wavelength=wavelength, flux=flux)
    with pytest.raises(ValueError):
        transform(z=np.array([0, -1, 2]), wavelength=wavelength, flux=flux)


def test_invalid_field_name():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    with pytest.raises(ValueError):
        transform(z=0, data_in=data_in, wavelength=np.arange(10))
    with pytest.raises(ValueError):
        transform(z=0, data_in=data_in, wavelength=123)


def test_missing_input_field():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    with pytest.raises(ValueError):
        transform(z=0, data_in=data_in, wavelength='wavelength')


def test_missing_output_field():
    data_in = np.empty((10,), dtype=[('wavelength', np.float32), ('flux', np.float64)])
    data_out = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    with pytest.raises(ValueError):
        transform(z=0, data_in=data_in, data_out=data_out)


def test_different_shapes():
    wavelength = np.arange(10)
    flux = np.zeros(shape=(11,))
    with pytest.raises(ValueError):
        transform(z=0, wavelength=wavelength, flux=flux)


def test_separate_arrays():
    wavelength = np.arange(10, dtype=np.float32)
    flux = np.zeros(shape=(10,), dtype=np.float64)
    result = transform(z=0, wavelength=wavelength, flux=flux)
    assert result.shape == (10,), 'Unexpected output array shape.'
    assert result.dtype == [('wavelength', np.float32), ('flux', np.float64)],\
        'Unexpected output array dtype.'
    assert np.allclose(wavelength, result['wavelength']), 'Invalid wavelength result.'
    assert np.allclose(flux, result['flux']), 'Invalid flux result.'


def test_data_in():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    data_in['wlen'] = np.arange(10)
    data_in['flux'] = 0.
    result = transform(z=0, data_in=data_in, wavelength='wlen')
    assert result.shape == (10,), 'Unexpected output array shape.'
    assert result.dtype == [('wlen', np.float32), ('flux', np.float64)],\
        'Unexpected output array dtype.'
    assert np.allclose(data_in['wlen'], result['wlen']), 'Invalid wavelength result.'
    assert np.allclose(data_in['flux'], result['flux']), 'Invalid flux result.'
    assert result.base is None, 'Result does not own its memory.'


def test_invalid_out_shape():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    data_out = np.empty((2, 5,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    with pytest.raises(ValueError):
        transform(z=0, data_in=data_in, data_out=data_out)


def test_data_in_different_out():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    data_in['wlen'] = np.arange(10)
    data_in['flux'] = 0.
    data_out = np.copy(data_in)
    assert data_out.base is None, 'Copy does not own its memory.'
    result = transform(z=0, data_in=data_in, data_out=data_out, wavelength='wlen')
    assert result is data_out, 'result is not equal to data_out.'
    assert data_out.shape == (10,), 'Unexpected output array shape.'
    assert data_out.dtype == [('wlen', np.float32), ('flux', np.float64)],\
        'Unexpected output array dtype.'
    assert np.allclose(data_in['wlen'], data_out['wlen']), 'Invalid wavelength result.'
    assert np.allclose(data_in['flux'], data_out['flux']), 'Invalid flux result.'
    assert data_out.base is None, 'result does not own its memory.'


def test_data_in_same_out():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    data_in['wlen'] = np.arange(10)
    data_in['flux'] = 0.
    data_out = transform(z=0, data_in=data_in, data_out=data_in, wavelength='wlen')
    assert data_out is data_in, 'data_out is not equal to data_in.'


def test_round_trip():
    wavelength = np.arange(10)
    flux = np.arange(10)
    result = transform(z=1, to_rest_frame=False, wavelength=wavelength, flux=flux)
    result = transform(z=1, to_rest_frame=True, data_in=result)
    assert np.allclose(wavelength, result['wavelength']),\
        'Wavelength arrays not equal after round trip.'
    assert np.allclose(flux, result['flux']),\
        'Flux arrays not equal after round trip.'


def test_multiple_spectra():
    data_in = np.empty((2, 10), dtype=[('wlen', np.float32), ('flux', np.float64)])
    result = transform(z=0, data_in=data_in, wavelength='wlen')
    assert np.allclose(data_in['wlen'], result['wlen']), 'Invalid wavelength result.'
    assert np.allclose(data_in['flux'], result['flux']), 'Invalid flux result.'


def test_broadcast_error():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    z = np.zeros(shape=(2,))
    with pytest.raises(ValueError):
        transform(z=z, data_in=data_in, wavelength='wlen')


def test_multiple_z():
    data_in = np.empty((10,), dtype=[('wlen', np.float32), ('flux', np.float64)])
    z = np.zeros(shape=(2, 3, 1))
    result = transform(z=z, data_in=data_in, wavelength='wlen')
    assert result.shape == (2, 3, 10), 'Invalid output shape.'
    assert np.allclose(data_in['wlen'], result['wlen'][1, 0]), 'Invalid wavelength result.'
    assert np.allclose(data_in['flux'], result['flux'][0, 2]), 'Invalid flux result.'
