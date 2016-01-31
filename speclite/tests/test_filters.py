# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..filters import *

import numpy as np
import math

import astropy.table
import astropy.units as u

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


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


def test_validate_units():
    validate_wavelength_array([1.])
    validate_wavelength_array([1.] * u.m)


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
    add_units = lambda fval: (lambda wlen: fval(wlen) * u.erg)
    wlen = np.arange(1, 3) * u.Angstrom
    for v in True, False:
        # Test each mode without any function return units.
        f1 = tabulate_function_of_wavelength(scalar_no_units, wlen, v)
        assert f1[1] == None
        f2 = tabulate_function_of_wavelength(scalar_with_units, wlen, v)
        assert f2[1] == None
        f3 = tabulate_function_of_wavelength(array_no_units, wlen, v)
        assert f3[1] == None
        f4 = tabulate_function_of_wavelength(scalar_with_units, wlen, v)
        assert f4[1] == None
        # Now test with return units.
        g1 = tabulate_function_of_wavelength(
            add_units(scalar_no_units), wlen, v)
        assert np.array_equal(f1[0], g1[0]) and g1[1] == u.erg
        g2 = tabulate_function_of_wavelength(
            add_units(scalar_with_units), wlen, v)
        assert np.array_equal(f2[0], g2[0]) and g2[1] == u.erg
        g3 = tabulate_function_of_wavelength(
            add_units(array_no_units), wlen, v)
        assert np.array_equal(f3[0], g3[0]) and g3[1] == u.erg
        g4 = tabulate_function_of_wavelength(
            add_units(scalar_with_units), wlen, v)
        assert np.array_equal(f4[0], g4[0]) and g4[1] == u.erg


def test_tabulate_not_func():
    wlen = np.arange(1, 3) * u.Angstrom
    for v in True, False:
        with pytest.raises(ValueError):
            tabulate_function_of_wavelength('not a function', wlen, v)


def test_tabulate_changing_units():
    wlen = [1, 2] * u.Angstrom
    f = lambda wlen: 1 * u.erg ** math.sqrt(wlen)
    verbose = True
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)
    f = lambda wlen: 1 * u.erg ** math.sqrt(wlen.value)
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)
    f = lambda wlen: 1 if wlen < 1.5 else 1 * u.erg
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)
    f = lambda wlen: 1 if wlen.value < 1.5 else 1 * u.erg
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)
    f = lambda wlen: 1 if wlen > 1.5 else 1 * u.erg
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)
    f = lambda wlen: 1 if wlen.value > 1.5 else 1 * u.erg
    with pytest.raises(RuntimeError):
        tabulate_function_of_wavelength(f, wlen, verbose)


def test_response():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    FilterResponse(wlen, [0, 1, 0], meta)
    FilterResponse(wlen, [0, 1, 0] * u.dimensionless_unscaled, meta)


def test_response_call():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    result = r(1.)
    result = r(1. * u.Angstrom)
    result = r(1. * u.micron)
    result = r([1.])
    result = r([1.] * u.Angstrom)
    result = r([1.] * u.micron)
    with pytest.raises(u.UnitConversionError):
        result = r(1. * u.erg)


def test_response_bad():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    with pytest.raises(ValueError):
        FilterResponse(wlen, [1, 2], meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [1, 2, 3] * u.erg, meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [0, -1, 0], meta)
    with pytest.raises(ValueError):
        FilterResponse(wlen, [0, 0, 0], meta)
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


def test_response_convolve():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    r.convolve_with_array([1, 3], [1, 1], interpolate=True)


def test_response_convolve_with_function():
    wlen = [1, 2, 3]
    resp = [0, 1, 0]
    meta = dict(group_name='', band_name='')
    filt = FilterResponse(wlen, resp, meta)
    filt.convolve_with_function(lambda wlen: 1.)
    with pytest.raises(ValueError):
        filt.convolve_with_function(lambda wlen: 1., method='none')


def test_response_mag():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    r.get_ab_maggies(lambda wlen: 1 * default_flux_unit)
    r.get_ab_maggies([1, 1] * default_flux_unit, [1, 3])
    r.get_ab_maggies([1, 1] * default_flux_unit,
                     [1, 3] * default_wavelength_unit)
    r.get_ab_magnitude(lambda wlen: 1 * default_flux_unit)
    r.get_ab_magnitude([1, 1] * default_flux_unit, [1, 3])
    r.get_ab_magnitude([1, 1] * default_flux_unit,
                       [1, 3] * default_wavelength_unit)


def test_response_bad_save():
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    with pytest.raises(ValueError):
        r.save('no such directory')


def test_response_save_load(tmpdir):
    wlen = [1, 2, 3]
    meta = dict(group_name='', band_name='')
    r1 = FilterResponse(wlen, [0, 1, 0], meta)
    save_name = r1.save(str(tmpdir))
    r2 = load_filter(save_name)
    assert np.array_equal(r1.wavelength, r2.wavelength)
    assert np.array_equal(r1.response, r2.response)
    assert r1.meta == r2.meta


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
    conv([1, 1] * u.erg)
    conv([[1, 1], [1, 1]])
    conv([[1, 1], [1, 1]] * u.erg)
    with pytest.raises(ValueError):
        conv([1, 1], method='none')
    with pytest.raises(ValueError):
        conv([1, 1, 1])


def test_convolution_plot():
    conv = FilterConvolution('sdss2010-r', [4000., 8000.], interpolate=True)
    conv([1, 1], plot=True)
    with pytest.raises(ValueError):
        conv([[1, 1], [1, 1]], plot=True)


def test_load_filter():
    load_filter('sdss2010-r', load_from_cache=False, verbose=True)
    load_filter('sdss2010-r', load_from_cache=True, verbose=True)
    with pytest.raises(ValueError):
        load_filter('none')


def test_load_bad(tmpdir):
    meta = dict(group_name='', band_name='')
    # Missing wavelength column.
    table = astropy.table.QTable(meta=meta)
    table['response'] = [1, 1]
    name = str(tmpdir.join('bad.ecsv'))
    table.write(name, format='ascii.ecsv')
    with pytest.raises(RuntimeError):
        load_filter(name)
    # Missing response column.
    table = astropy.table.QTable(meta=meta)
    table['wavelength'] = [1, 2] * u.Angstrom
    name = str(tmpdir.join('bad.ecsv'))
    table.write(name, format='ascii.ecsv')
    with pytest.raises(RuntimeError):
        load_filter(name)
    # Missing wavelength units.
    table = astropy.table.QTable(meta=meta)
    table['wavelength'] = [1, 2]
    table['response'] = [1, 1]
    name = str(tmpdir.join('bad.ecsv'))
    table.write(name, format='ascii.ecsv')
    with pytest.raises(RuntimeError):
        load_filter(name)
    # Unexpected response units.
    table = astropy.table.QTable(meta=meta)
    table['wavelength'] = [1, 2] * u.Angstrom
    table['response'] = [1, 1] * u.erg
    name = str(tmpdir.join('bad.ecsv'))
    table.write(name, format='ascii.ecsv')
    with pytest.raises(RuntimeError):
        load_filter(name)


def test_load_filter_group():
    load_filter_group('sdss2010')


def test_plot_filters():
    plot_filters('sdss2010')
    plot_filters('sdss2010', ['g'])
    plot_filters('sdss2010', 'g')
    plot_filters('sdss2010', ['g', 'r'])
    plot_filters('sdss2010', 'gr')
    plot_filters(names = ['sdss2010-g', 'sdss2010-r'])
    plot_filters('sdss2010', 'g', legend_loc=None)


def test_plot_filters_bad():
    with pytest.raises(ValueError):
        plot_filters()
    with pytest.raises(ValueError):
        plot_filters(names=123)
    with pytest.raises(ValueError):
        plot_filters('no such group')
    with pytest.raises(ValueError):
        plot_filters('sdss2010', ['x'])
    with pytest.raises(ValueError):
        plot_filters('sdss2010', 'g', wavelength_unit=u.erg)


def test_plot_filters_limits():
    plot_filters(names=['sdss2010-r'], wavelength_limits=(1,2))
    plot_filters(names=['sdss2010-r'], wavelength_limits=(1*u.m,2*u.m))
    plot_filters(names=['sdss2010-r'], wavelength_limits=(1,2*u.m))
    plot_filters(names=['sdss2010-r'], wavelength_limits=(1*u.m,2))
    with pytest.raises(ValueError):
        plot_filters(names=['sdss2010-r'], wavelength_limits='bad limits')
    with pytest.raises(ValueError):
        plot_filters(names=['sdss2010-r'], wavelength_limits=(1,))
    with pytest.raises(ValueError):
        plot_filters(names=['sdss2010-r'], wavelength_limits=(1,2,3))
    with pytest.raises(ValueError):
        plot_filters(names=['sdss2010-r'], wavelength_limits=(1*u.erg,2))
    with pytest.raises(ValueError):
        plot_filters(names=['sdss2010-r'], wavelength_limits=(1,2*u.erg))