# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..filters import *

import numpy as np
import math

import astropy.units as u


def test_ab_invalid_wlen():
    with pytest.raises(ValueError):
        ab_reference_flux(1 * u.s)


def test_validate_bad_wlen():
    with pytest.raises(ValueError):
        validate_wavelength_array(1. * u.Angstrom)
    with pytest.raises(ValueError):
        validate_wavelength_array([[1.]] * u.Angstrom)
    with pytest.raises(ValueError):
        validate_wavelength_array([1.] * u.Angstrom, min_length=2)
    with pytest.raises(ValueError):
        validate_wavelength_array([2., 1.] * u.Angstrom)


def test_validate_default_units():
    validate_wavelength_array([1.])


def test_tabulate_wlen_units():
    with pytest.raises(ValueError):
        tabulate_function_of_wavelength(lambda wlen: 1, [1.])
    with pytest.raises(ValueError):
        tabulate_function_of_wavelength(lambda wlen: 1, [1.] * u.s)


def test_tabulate():
    scalar_no_units = lambda wlen: math.sqrt(wlen)
    scalar_with_units = lambda wlen: math.sqrt(wlen.value)
    array_no_units = lambda wlen: 1. + np.sqrt(wlen)
    array_with_units = lambda wlen: np.sqrt(wlen.value)
    wlen = np.arange(1, 3) * u.Angstrom
    # Test each mode without any function return units.
    f1 = tabulate_function_of_wavelength(scalar_no_units, wlen)
    assert f1[1] == None
    f2 = tabulate_function_of_wavelength(scalar_with_units, wlen)
    assert f2[1] == None
    f3 = tabulate_function_of_wavelength(array_no_units, wlen)
    assert f3[1] == None
    f4 = tabulate_function_of_wavelength(scalar_with_units, wlen)
    assert f4[1] == None
    # Now test with return units.
    add_units = lambda fval: (lambda wlen: fval(wlen) * u.erg)
    g1 = tabulate_function_of_wavelength(add_units(scalar_no_units), wlen)
    assert np.array_equal(f1[0], g1[0]) and g1[1] == u.erg
    g2 = tabulate_function_of_wavelength(add_units(scalar_with_units), wlen)
    assert np.array_equal(f2[0], g2[0]) and g2[1] == u.erg
    g3 = tabulate_function_of_wavelength(add_units(array_no_units), wlen)
    assert np.array_equal(f3[0], g3[0]) and g3[1] == u.erg
    g4 = tabulate_function_of_wavelength(add_units(scalar_with_units), wlen)
    assert np.array_equal(f4[0], g4[0]) and g4[1] == u.erg


def test_tabulate_changing_units():
    wlen = [1, 2] * u.Angstrom
    f = lambda wlen: 1 * u.erg ** math.sqrt(wlen)
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose=True)
    f = lambda wlen: 1 * u.erg ** math.sqrt(wlen.value)
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose=True)


def test_response_bad_response():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    FilterResponse(wlen, [0, 1, 0], meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [1, 2], meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [1, 2] * u.erg, meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, np.zeros_like(wlen), meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [1, 1, 0], meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [0, 1, 1], meta)


def test_response_trim():
    wlen = [1, 2, 3, 4, 5]
    meta = dict(group_name='', band_name='')
    assert np.array_equal(
        FilterResponse(wlen, [0, 0, 1, 1, 0], meta).wavelength, [2, 3, 4, 5])
    assert np.array_equal(
        FilterResponse(wlen, [0, 1, 1, 0, 0], meta).wavelength, [1, 2, 3, 4])
    assert np.array_equal(
        FilterResponse(wlen, [0, 0, 1, 0, 0], meta).wavelength, [2, 3, 4])


def test_response_bad_meta():
    wlen = [1, 2, 3]
    resp = [0, 1, 0]
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict())
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name=''))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(band_name=''))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, 123)


def test_response_convolve_with_function():
    wlen = [1, 2, 3]
    resp = [0, 1, 0]
    meta = dict(group_name='', band_name='')
    filt = FilterResponse(wlen, resp, meta)
    filt.convolve_with_function(lambda wlen: 1.)
    with pytest.raises(ValueError):
        filt.convolve_with_function(lambda wlen: 1., method='none')


def test_convolution_ctor():
    FilterConvolution('sdss2010-r', [4000., 8000.], interpolate=True)
    rband = load_filter('sdss2010-r')
    FilterConvolution(rband, [4000., 8000.], interpolate=True)
    FilterConvolution(rband, np.arange(4000, 8000, 5))
    FilterConvolution(rband, np.arange(4000, 8000, 5), photon_weighted=False)
    with pytest.raises(ValueError):
        FilterConvolution(
            rband, [4000., 8000.], interpolate=False)
    with pytest.raises(ValueError):
        FilterConvolution(rband, [5000., 6000.])


def test_convolution_call():
    conv = FilterConvolution('sdss2010-r', [4000., 8000.], interpolate=True)
    conv([1, 1])
    with pytest.raises(ValueError):
        conv([1, 1], method='none')
    with pytest.raises(ValueError):
        conv([1, 1, 1], method='none')


def test_convolution_plot():
    # Not sure how to do this since it opens a matplotlib window.
    pass


def test_load_filter():
    load_filter('sdss2010-r', verbose=True) # First time to load cache
    load_filter('sdss2010-r', verbose=True) # Second time to fetch from cache
    with pytest.raises(ValueError):
        load_filter('none')


def test_load_bad_file():
    # This requires writing some temporary files with bad formats then
    # cleaning up afterwards.  Should add support for reading from a
    # directory other than <pkg>/data/filters/ for this.
    pass


def plot_filters():
    # How to do this?
    pass
