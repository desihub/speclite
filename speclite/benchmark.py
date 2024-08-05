# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Driver routines for benchmarking and profiling.
"""
from __future__ import print_function, division

import time

import numpy as np

import astropy.table
import astropy.units as u
import argparse

import speclite.filters


def magnitude_calculation(results, num_repeats):
    """Run a suite of magnitude calclulations.

    Parameters
    ----------
    results : astropy.table.Table
        Table where results should be appended.
    num_repeats : int
        Number of times to repeat the timing loop.
    """
    # Initialize a flat spectrum with 1A binning.
    wlen = np.arange(3500., 10500., 1.) * u.Angstrom
    flux_unit = u.erg / u.cm**2 / u.s / u.Angstrom
    flux = np.ones_like(wlen.value) * 1e-17 * flux_unit
    # Load a filter.
    rband = speclite.filters.load_filter('sdss2010-r')

    start = time.time()
    for i in range(num_repeats):
        m = rband.get_ab_maggies(flux, wlen)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'get_ab_maggies', timing))

    start = time.time()
    for i in range(num_repeats):
        m = rband.get_ab_maggies(flux.value, wlen.value)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'get_ab_maggies (value)', timing))

    start = time.time()
    for i in range(num_repeats):
        convolution = rband.convolve_with_array(
            wlen, flux, interpolate=True, photon_weighted=True,
            axis=-1, units=flux_unit)
        m = convolution.value / rband.ab_zeropoint.value
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'convolve_with_array (units)', timing))

    start = time.time()
    for i in range(num_repeats):
        m = rband.convolve_with_array(
            wlen, flux, interpolate=True, units=flux_unit)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'convolve_with_array (no units)', timing))

    start = time.time()
    for i in range(num_repeats):
        conv = speclite.filters.FilterConvolution(
            rband, wlen, interpolate=True, units=flux_unit)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'FilterConvolution ctor', timing))

    conv = speclite.filters.FilterConvolution(
        rband, wlen, interpolate=True, units=flux_unit)
    start = time.time()
    for i in range(num_repeats):
        m = conv(flux)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'FilterConvolution __call__', timing))

    spectra = np.ones((num_repeats, len(wlen))) * flux_unit
    start = time.time()
    m = rband.get_ab_maggies(spectra, wlen)
    timing = 1e6 * (time.time() - start) / num_repeats
    assert m.shape == (num_repeats,)
    results.add_row(('filters', 'Multidim. array', timing))

    return results


def main(argv=None):
    """Entry-point for :command:`speclite_benchmark`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num-repeats', type=int, default=1000,
        help = 'number of times to repeat timing loops')
    parser.add_argument('-a', '--all', action='store_true',
        help = 'run all benchmark suites')
    parser.add_argument('-m', '--magnitude', action='store_true',
        help = 'benchmark magnitude calculations')
    parser.add_argument('-s', '--save', type=str, default=None,
        help='Name of file to save results to (or print if not set)')
    parser.add_argument('-f', '--format', type=str,
        default='ascii.fixed_width_two_line',
        help='format to use for results')
    args = parser.parse_args(argv)

    results = astropy.table.Table(
        names=('Suite', 'Description', 'Time [us]'),
        dtype=('S8', 'S40', float))
    if args.magnitude or args.all:
        results = magnitude_calculation(results, args.num_repeats)
    else:
        print('ERROR: No test suite specified (--magnitude or --all)!')
        return 1

    results.write(args.save, format=args.format,
                  delimiter_pad=' ', position_char='=',
                  formats={'Time [us]': '%.1f'})
    return 0
