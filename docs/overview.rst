Overview
========

Speclite is a lightweight package for performing basic operations on spectral
data contained in numpy arrays.  The basic philosophy of this package is to:

 * Use spectral data in numpy arrays "as-is" rather than requiring users to reformat their data or wrap it in objects.
 * Minimize the number of assumptions about what quantities are used to define a spectrum. For example, we do not assume that spectra are sampled in wavelength rather than frequency.
 * Support operations on `masked arrays <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ and correctly propagate the effects of invalid samples.
 * Use per-sample weights (inverse variances) when these are available.
 * Do not introduce any dependencies beyond `numpy <http://www.numpy.org/>`__, `scipy <http://www.scipy.org/>`__, and `astropy <http://astropy.readthedocs.io/>`__.
 * Be compatible with python versions 2.6, 2.7, 3.3, 3.4 and numpy versions >= 1.8.
 * Be fully documented and unit tested, with 100% test coverage.
 * Use the `astropy affiliated package template <https://github.com/astropy/package-template>`__ to benefit from its sophisticated configuration and integration with other services (TravisCI, coveralls.io, readthedocs.io) and for possible future distribution with astropy.

Speclite provides functions to perform the following basic manipulations of spectroscopic data:

 * :func:`redshift() <speclite.redshift>`: transforms from one redshift to another.
 * :func:`resample() <speclite.resample>`: resamples from one sampling grid to another using interpolation.
 * :func:`downsample() <speclite.downsample>`: downsamples by combining bins in consecutive groups.
 * :func:`accumulate() <speclite.accumulate>`: combines two spectra on the same grid, and can efficiently stack many spectra.
 * :mod:`filters module <speclite.filters>`: convolutions and magnitude calculations for some :doc:`reference filters <filters>`.

Operations that are planned for future versions include:

 * Signal-to-noise estimation.

The results of the redshift, resample, downsample and accumulate operations
are always numpy `structured arrays
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ and these operations
also accept structured arrays these as inputs.  Some operations also accept
un-structured arrays as inputs, but structured arrays should generally be
preferred since they have negligible overhead and significantly reduce the
burden on users to keep track of which columns correspond to which quantities.

In cases where the source data is not already in a structured array, the
necessary metadata can be added without copying the underlying array data,
for example:

    >>> import numpy as np
    >>> data = np.arange(9, dtype=float).reshape(3, 3)
    >>> data
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])
    >>> sdata = data.view([('wlen', float), ('flux', float),
    ... ('ivar', float)]).reshape(3)
    >>> sdata
    array([(0.0, 1.0, 2.0), (3.0, 4.0, 5.0), (6.0, 7.0, 8.0)],
          dtype=[('wlen', '<f8'), ('flux', '<f8'), ('ivar', '<f8')])
    >>> sdata.base is data.base
    True

You should normally use dictionary notation to refer to individual fields of
a structured array:

    >>> sdata['wlen']
    array([ 0.,  3.,  6.])

You can also use the more convenient dot notation using `this recarray recipe
<http://wiki.scipy.org/Cookbook/Recarray>`__. However, this is generally
slower so speclite results are not automatically converted to `numpy recarrays
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`__.

`Examples of using speclite
<https://github.com/dkirkby/bossdata/blob/master/examples/nb/
StackingWithSpeclite.ipynb>`__
with BOSS data are included with the `bossdata
<https://bossdata.readthedocs.io/en/latest/>`__ package.
