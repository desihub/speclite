# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Driver routines for benchmarking and profiling.
"""
from __future__ import print_function, division

import time

import numpy as np

import astropy.table
import astropy.units as u
from astropy.utils.compat import argparse

import speclite


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
    for i in xrange(num_repeats):
        m = rband.get_ab_maggies(flux, wlen)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'Init each time', timing))

    start = time.time()
    for i in xrange(num_repeats):
        m = rband.get_ab_maggies(flux.value, wlen.value)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'No units', timing))

    start = time.time()
    conv = speclite.filters.FilterConvolution(rband, wlen)
    for i in xrange(num_repeats):
        m = conv(flux)
    timing = 1e6 * (time.time() - start) / num_repeats
    results.add_row(('filters', 'First time only', timing))

    spectra = np.ones((num_repeats, len(wlen))) * flux_unit
    start = time.time()
    m = rband.get_ab_maggies(spectra, wlen)
    timing = 1e6 * (time.time() - start) / num_repeats
    assert m.shape == (num_repeats,)
    results.add_row(('filters', 'Multidim. array', timing))

    return results


def main(argv=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num-repeats', type=int, default=1000,
        help = 'number of times to repeat timing loops')
    parser.add_argument('--magnitude', action='store_true',
        help = 'benchmark magnitude calculations')
    parser.add_argument('--save', type=str, default=None,
        help='Name of file to save results to (or print if not set)')
    parser.add_argument('--format', type=str,
        default='ascii.fixed_width_two_line',
        help='format to use for results')
    args = parser.parse_args(argv)

    results = astropy.table.Table(
        names=('Suite', 'Description', 'Timing'),
        dtype=('S8', 'S20', float))
    if args.magnitude:
        results = magnitude_calculation(results, args.num_repeats)

    results.write(args.save, format=args.format,
                  delimiter_pad=' ', position_char='=',
                  formats={'Timing': '%.1f'})
    return 0


if __name__ == '__main__':
    main()
