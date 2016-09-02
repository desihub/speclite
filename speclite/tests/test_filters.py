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


def test_validate_array():
    wave = np.arange(1000., 6000., 100.)
    wave2 = validate_wavelength_array(wave)
    assert wave is wave2
    validate_wavelength_array(wave + 500.)
    validate_wavelength_array(wave - 500.)


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
    meta = dict(group_name='g', band_name='b')
    FilterResponse(wlen, [0, 1, 0], meta)
    FilterResponse(wlen, [0, 1, 0] * u.dimensionless_unscaled, meta)


def test_response_call():
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
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
    meta = dict(group_name='g', band_name='b')
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


def test_response_band_shift():
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
    r0 = FilterResponse(wlen, [0, 1, 0], meta)
    r1 = FilterResponse(wlen, [0, 1, 0], meta, band_shift=1)
    r2 = r0.create_shifted(band_shift=1)
    assert np.array_equal(r1._wavelength, r0._wavelength / 2)
    assert np.array_equal(r1._wavelength, r2._wavelength)
    assert np.array_equal(r1.response, r0.response)
    assert np.array_equal(r2.response, r0.response)
    with pytest.raises(ValueError):
        r = FilterResponse(wlen, [0, 1, 0], meta, band_shift=-1)
    with pytest.raises(RuntimeError):
        r1.save()
    with pytest.raises(RuntimeError):
        r2.create_shifted(1)


def test_response_trim():
    wlen = [1, 2, 3, 4, 5]
    meta = dict(group_name='g', band_name='b')
    assert np.array_equal(
        FilterResponse(wlen, [0, 0, 1, 1, 0], meta)._wavelength, [2, 3, 4, 5])
    assert np.array_equal(
        FilterResponse(wlen, [0, 1, 1, 0, 0], meta)._wavelength, [1, 2, 3, 4])
    assert np.array_equal(
        FilterResponse(wlen, [0, 0, 1, 0, 0], meta)._wavelength, [2, 3, 4])


def test_response_bad_meta():
    wlen = [1, 2, 3]
    resp = [0, 1, 0]
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, 123)
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict())
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name='g'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(band_name='b'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name=123, band_name='b'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name='0', band_name='b'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name='g-*', band_name='b'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name='g', band_name='b.ecsv'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name='g\n', band_name='b'))
    with pytest.raises(ValueError):
        FilterResponse(wlen, resp, dict(group_name=' g', band_name='b'))


def test_response_convolve():
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    r.convolve_with_array([1, 3], [1, 1], interpolate=True)


def test_response_convolve_with_function():
    wlen = [1, 2, 3]
    resp = [0, 1, 0]
    meta = dict(group_name='g', band_name='b')
    filt = FilterResponse(wlen, resp, meta)
    filt.convolve_with_function(lambda wlen: 1.)
    filt.convolve_with_function(lambda wlen: 1. * u.erg)
    filt.convolve_with_function(lambda wlen: 1. * u.erg, units=u.erg)
    filt.convolve_with_function(lambda wlen: 1., units=u.erg)
    with pytest.raises(ValueError):
        filt.convolve_with_function(lambda wlen: 1., method='none')
    with pytest.raises(ValueError):
        filt.convolve_with_function(lambda wlen: 1. * u.m, units=u.erg)

def test_response_mag():
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    r.get_ab_maggies(lambda wlen: 1.)
    r.get_ab_maggies(lambda wlen: 1. * default_flux_unit)
    r.get_ab_maggies([1., 1.], [1, 3])
    r.get_ab_maggies([1, 1] * default_flux_unit, [1, 3])
    r.get_ab_maggies([1, 1] * default_flux_unit,
                     [1, 3] * default_wavelength_unit)
    r.get_ab_magnitude(lambda wlen: 1 * default_flux_unit)
    r.get_ab_magnitude([1, 1] * default_flux_unit, [1, 3])
    r.get_ab_magnitude([1, 1] * default_flux_unit,
                       [1, 3] * default_wavelength_unit)


def test_mag_wavelength_units():
    # Check that non-default wavelength units are handled correctly.
    wlen = [1, 2, 3] * u.Angstrom
    meta = dict(group_name='g', band_name='b')
    r = FilterResponse(wlen, [0, 1, 0], meta)

    # Note that some margin is required to allow for roundoff error
    # when converting wlen to the default units.
    eps = 1e-6
    wlen = [0.1 - eps, 0.3 + eps] * u.nm
    flux = [1., 1.] * default_flux_unit
    m1 = r.get_ab_maggies(flux, wlen)
    m2 = r.get_ab_maggies(flux, wlen)
    assert m1 == m2


def test_mag_flux_units():
    # Check that non-default flux units are handled correctly.
    wlen = [1, 2, 3] * u.Angstrom
    meta = dict(group_name='g', band_name='b')
    r = FilterResponse(wlen, [0, 1, 0], meta)

    flux = lambda w: 1. * u.W / (u.m**2 * u.micron)
    m1 = r.get_ab_maggies(flux)
    m2 = r.get_ab_maggies(flux)
    assert m1 == m2

    wlen = [1., 3.] * default_wavelength_unit
    flux = [1., 1.] * u.W / (u.m**2 * u.micron)
    m1 = r.get_ab_maggies(flux, wlen)
    m2 = r.get_ab_maggies(flux, wlen)
    assert m1 == m2


def test_response_bad_save():
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
    r = FilterResponse(wlen, [0, 1, 0], meta)
    with pytest.raises(ValueError):
        r.save('no such directory')


def test_response_save_load(tmpdir):
    wlen = [1, 2, 3]
    meta = dict(group_name='g', band_name='b')
    r1 = FilterResponse(wlen, [0, 1, 0], meta)
    save_name = r1.save(str(tmpdir))
    r2 = load_filter(save_name)
    assert np.array_equal(r1._wavelength, r2._wavelength)
    assert np.array_equal(r1.response, r2.response)
    assert r1.meta == r2.meta


def test_convolution_ctor():
    FilterConvolution('sdss2010-r', [4000., 8000.], interpolate=True)
    rband = load_filter('sdss2010-r')
    FilterConvolution(rband, [4000., 8000.], interpolate=True)
    FilterConvolution(rband, np.arange(4000, 8000, 5))
    FilterConvolution(rband, np.arange(4000, 8000, 5), photon_weighted=False)
    FilterConvolution(rband, np.arange(4000, 8000, 5),
                      photon_weighted=False, units=default_flux_unit)
    with pytest.raises(ValueError):
        FilterConvolution(
            rband, [4000., 8000.], interpolate=False)
    with pytest.raises(ValueError):
        FilterConvolution(rband, [5000., 6000.])


def test_convolution_call():
    conv = FilterConvolution('sdss2010-r', [4000., 8000.],
                             interpolate=True, units=u.erg)
    conv([1, 1])
    conv([1, 1] * u.erg)
    conv([[1, 1], [1, 1]])
    conv([[1, 1], [1, 1]] * u.erg)
    conv([[1, 1], [1, 1]] * u.erg)
    with pytest.raises(ValueError):
        conv([1, 1], method='none')
    with pytest.raises(ValueError):
        conv([1, 1, 1])
    with pytest.raises(ValueError):
        conv([[1, 1], [1, 1]] * u.m)


def test_convolution_call_no_units():
    conv = FilterConvolution('sdss2010-r', [4000., 8000.],
                             interpolate=True, units=None)
    with pytest.raises(ValueError):
        conv([[1, 1], [1, 1]] * default_flux_unit)


def test_response_pad_methods():
    rband = load_filter('sdss2010-r')
    wave = rband._wavelength - 500.
    flux = np.ones_like(wave)
    for method in FilterResponse._pad_methods:
        pwave, pflux = rband.pad_spectrum(flux, wave, method=method)
    with pytest.raises(ValueError):
        pwave, pflux = rband.pad_spectrum(flux, wave, method='none')


def test_response_pad_units():
    rband = load_filter('sdss2010-r')
    wave = rband._wavelength - 500.
    flux = np.ones_like(wave)
    # No units.
    flux, wave = rband.pad_spectrum(flux, wave)
    with pytest.raises(AttributeError):
        flux.unit
    with pytest.raises(AttributeError):
        wave.unit
    # Wavelength units.
    wave = ((rband._wavelength - 500.) * u.Angstrom).to(u.micron)
    flux = np.ones_like(wave.value)
    flux, wave = rband.pad_spectrum(flux, wave)
    with pytest.raises(AttributeError):
        flux.unit
    assert wave.unit == u.micron
    # Flux units.
    wave = rband._wavelength - 500.
    flux = np.ones_like(wave) * default_flux_unit
    flux, wave = rband.pad_spectrum(flux, wave)
    assert flux.unit == default_flux_unit
    with pytest.raises(AttributeError):
        wave.unit


def test_response_pad_ranges():
    rband = load_filter('sdss2010-r')
    wave = rband._wavelength[:]
    flux = np.ones_like(wave)
    # No pad necessary.
    pflux, pwave = rband.pad_spectrum(flux, wave)
    rband.get_ab_maggies(pflux, pwave)
    # Pad above only.
    pflux, pwave = rband.pad_spectrum(flux, wave + 500.)
    rband.get_ab_maggies(pflux, pwave)
    # Pad below only.
    pflux, pwave = rband.pad_spectrum(flux, wave - 500.)
    rband.get_ab_maggies(pflux, pwave)
    # Pad both sides.
    pflux, pwave = rband.pad_spectrum(flux[10:-10], wave[10:-10])
    rband.get_ab_maggies(pflux, pwave)


def test_response_pad_shape():
    rband = load_filter('sdss2010-r')
    wave = np.linspace(5000., 10000., 100)
    for shape, axis in (
        ((100,), -1), ((100,), 0),
        ((2, 100), 1), ((100, 2), 0),
        ((2, 3, 100), 2), ((2, 100, 3), 1), ((100, 2, 3), 0)):
        flux = np.ones(shape=shape)
        pflux, pwave = rband.pad_spectrum(flux, wave, axis=axis)
        expected_shape = list(flux.shape)
        expected_shape[axis] = len(pwave)
        assert list(pflux.shape) == expected_shape


def test_sequence_pad():
    filters = load_filters('sdss2010-r', 'sdss2010-g')
    wave = np.linspace(5000., 10000., 100)
    flux = np.ones_like(wave)
    pflux, pwave = filters.pad_spectrum(flux, wave)
    filters.get_ab_maggies(pflux, pwave)


def test_load_none():
    filters = load_filters()
    assert len(filters) == 0
    assert np.array_equal(filters.effective_wavelengths.value, [])
    wave = np.linspace(5000., 10000., 100)
    flux = np.ones_like(wave)
    pflux, pwave = filters.pad_spectrum(flux, wave)
    assert wave is pwave
    assert flux is pflux


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
    with pytest.raises(ValueError):
        load_filter('none.dat')


def test_load_badname():
    with pytest.raises(ValueError):
        load_filters('sdss-*')
    with pytest.raises(ValueError):
        load_filters('decam2014*')
    with pytest.raises(ValueError):
        load_filters('nosuch')
    with pytest.raises(ValueError):
        load_filters('sdss-y')
    with pytest.raises(ValueError):
        load_filters('sdss2010-y')
    with pytest.raises(ValueError):
        load_filters('nosuch.ecsv')


def test_load_bad(tmpdir):
    meta = dict(group_name='g', band_name='b')
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


def test_response_sequence():
    s = load_filters('sdss2010-r')
    r = s[0]
    assert r in s
    assert s.names == [r.name]


def test_response_sequence_calls():
    s = load_filters('sdss2010-r')
    wlen = [2000, 12000]
    flux = [1, 1]
    t = s.get_ab_maggies(flux, wlen)
    assert t.colnames == s.names
    s.get_ab_magnitudes(flux, wlen)


def test_partial_coverage():
    sdss = load_filters('sdss2010-*')
    wlen = np.arange(5000, 10000) * u.Angstrom
    flux = np.ones_like(wlen.value) * u.erg / (u.cm**2 * u.s * u.Angstrom)
    maggies = sdss.get_ab_magnitudes(flux, wlen, mask_invalid=True)
    assert maggies.masked
    assert np.all(maggies['sdss2010-u'].mask)
    assert not np.any(maggies['sdss2010-r'].mask)
    mags = sdss.get_ab_magnitudes(flux, wlen, mask_invalid=True)
    assert mags.masked
    assert np.all(mags['sdss2010-u'].mask)
    assert not np.any(mags['sdss2010-r'].mask)
    with pytest.raises(ValueError):
        mags = sdss.get_ab_magnitudes(flux, wlen, mask_invalid=False)


def test_load_filters():
    load_filters('sdss2010-*')
    load_filters('sdss2010-r', 'wise2010-W4')


def test_plot_filters():
    plot_filters(load_filters('sdss2010-r'))
    plot_filters(load_filters('sdss2010-g', 'sdss2010-r'))


def test_plot_filters_bad():
    r = load_filters('sdss2010-g')
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_unit=u.erg)


def test_plot_filters_limits():
    r = load_filters('sdss2010-g')
    plot_filters(r, wavelength_limits=(1,2))
    plot_filters(r, wavelength_limits=(1*u.m,2*u.m))
    plot_filters(r, wavelength_limits=(1,2*u.m))
    plot_filters(r, wavelength_limits=(1*u.m,2))
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_limits='bad limits')
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_limits=(1,))
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_limits=(1,2,3))
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_limits=(1*u.erg,2))
    with pytest.raises(ValueError):
        plot_filters(r, wavelength_limits=(1,2*u.erg))


def test_explanatory_plot(tmpdir):
    save = str(tmpdir.join('sampling.png'))
    filter_sampling_explanatory_plot(save)


def test_benchmark():
    from speclite.benchmark import main
    main('-n 10 --all'.split())
