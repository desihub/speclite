# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Apply redshift transformations to wavelength, flux, inverse variance, etc.

Attributes
----------
exponents : dict

    Dictionary of predefined array names and corresponding redshift exponents,
    used by :func:`transform` to automatically select the correct exponent.
"""
from __future__ import print_function, division

import numpy as np
import numpy.ma as ma

import speclite.utility


exponents = dict(
    wlen=+1, wavelength=+1, wavelength_error=+1,
    freq=-1, frequency=-1, frequency_error=-1,
    flux=-1, irradiance_per_wavelength=-1,
    irradiance_per_frequency=+1,
    ivar=+2, ivar_irradiance_per_wavelength=+2,
    ivar_irradiance_per_frequency=-2)


def apply_redshift_transform(z_in, z_out, data_in, data_out, exponent):
    """Apply a redshift transform to spectroscopic quantities.

    The input redshifts can either be scalar values or arrays.  If either is
    an array, the result will be broadcast using their shapes.

    Parameters
    ----------
    z_in : float or numpy.ndarray
        Redshift(s) of the input spectral data, which must all be > -1.
    z_out : float or numpy.ndarray
        Redshift(s) of the output spectral data, which must all be > -1.
    data_in : dict
        Dictionary of numpy-compatible arrays of input quantities to transform.
        Usually obtained using :func:`speclite.utility.prepare_data`.
    data_out : dict
        Dictionary of numpy-compatible arrays to fill with transformed values.
        Usually obtained using :func:`speclite.utility.prepare_data`.
        The names used here must be a subset of the names appearing in
        data_in.
    exponent : dict
        Dictionary of exponents :math:`n` to use in the factor that
        transforms each input array:

        .. math::

            \left(\\frac{1 + z_{out}}{1 + z_{in}}\\right)^n

        Any names appearing in data_out that are not included here will be
        passed through unchanged.
    """
    # Calculate the redshift multiplicative factor, which might have a
    # non-trivial shape if either z_in or z_out is an array.
    z_in = np.asarray(z_in)
    z_out = np.asarray(z_out)
    if np.any(z_in <= -1):
        raise ValueError('Found invalid z_in <= -1.')
    if np.any(z_out <= -1):
        raise ValueError('Found invalid z_out <= -1.')
    zfactor = (1. + z_out) / (1. + z_in)

    # Fill data_out with transformed arrays.
    for name in data_out:
        n = exponent.get(name, 0)
        if n != 0:
            data_out[name][:] = data_in[name] * zfactor ** exponent[name]
        # This condition is not exhaustive but avoids an un-necessary copy
        # in the most comment case that data_out[name] is a direct view
        # of data_in[name].
        elif not (data_out[name].base is data_in[name]):
            data_out[name][:] = data_in[name]

    return data_out


def transform(*args, **kwargs):
    """Convenience method for performing a redshift transform.

    See :func:`apply_redshift_transform` for details. The exponents used
    to transform each input array are inferred from the array names,
    which must be listed in :data:`redshift.exponents`.

    >>> wlen0 = np.arange(4000., 10000.)
    >>> flux0 = np.ones(wlen0.shape)
    >>> result = transform(z_in=0, z_out=1, wlen=wlen0, flux=flux0)
    >>> wlen, flux = result['wlen'], result['flux']
    >>> flux[:5]
    array([ 0.5,  0.5,  0.5,  0.5,  0.5])

    Parameters
    ----------
    *args : list
        Arguments specifying the arrays to transform and passed to
        :func:`speclite.utility.prepare_data`.
    **kwargs : dict
        Arguments specifying the arrays to transform and passed to
        :func:`speclite.utility.prepare_data`, after filtering out the
        keywords listed below.
    z : float or numpy.ndarray or None
        Redshift(s) of the output spectra data, which must all be > -1.
        An input redshift of zero is assumed.  Cannot be combined with
        z_in or z_out.
    z_in : float or numpy.ndarray or None
        Redshift(s) of the input spectral data, which must all be > -1.
        When specified, z_out must also be specified. Cannot be combined
        with the z parameter.
    z_out : float or numpy.ndarray
        Redshift(s) of the output spectral data, which must all be > -1.
        When specified, z_in must also be specified. Cannot be combined
        with the z parameter.
    in_place : bool
        When True, the transform is performed in place, if possible, or
        a ValueError is raised.

    Returns
    -------
    tabular or dict
        A tabular object (astropy table or numpy structured array) matching
        the input type, or else a dictionary of numpy arrays if the input
        consists of arrays passed as keyword arguments.

    Raises
    ------
    ValueError
        Cannot perform redshift in place when broadcasting or invalid
        combination of z, z_in, z_out options.
    """
    kwargs, options = speclite.utility.get_options(
        kwargs, in_place=False, z_in=None, z_out=None, z=None)
    if options['z'] is not None:
        if options['z_in'] is not None or options['z_out'] is not None:
            raise ValueError('Cannot combine z parameter with z_in, z_out.')
        z_in = 0
        z_out = options['z']
    elif options['z_in'] is None or options['z_out'] is None:
        raise ValueError('Must both z_in and z_out.')
    else:
        z_in = options['z_in']
        z_out = options['z_out']

    # Prepare a read-only view of the input data.
    data_in, value = speclite.utility.prepare_data('read_only', args, kwargs)

    # Determine the output shape.
    input_array = data_in[data_in.keys()[0]]
    output_shape = np.broadcast(input_array, z_in, z_out).shape

    # Prepare the output arrays where the transformed results will be saved.
    if options['in_place']:
        if input_array.shape != output_shape:
            raise ValueError(
                'Cannot perform redshift in place when broadcasting.')
        data_out, return_value = speclite.utility.prepare_data(
            'in_place', args, kwargs)
    else:
        data_out, return_value = speclite.utility.prepare_data(
            output_shape, args, kwargs)

    # Determine the exponent to use for each array.
    exponent = {}
    global exponents
    for name in data_out:
        if name in exponents:
            exponent[name] = exponents[name]

    apply_redshift_transform(z_in, z_out, data_in, data_out, exponent)
    return return_value
