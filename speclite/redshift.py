import numpy as np


def transform(z, to_rest_frame=False, data_in=None, data_out=None,
              wavelength='wavelength', flux='flux'):
    """
    Apply a redshift transform to a spectrum.

    The input spectrum (or spectra) must have at least wavelength and flux values,
    which are transformed according to::

        wavelength_out = wavelength_in * (1 + z)
        flux_out = flux_in / (1 + z)

    or, if ``to_rest_frame`` is True::

        wavelength_out = wavelength_in / (1 + z)
        flux_out = flux_in * (1 + z)

    The usual `numpy broadcasting rules <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ apply so that the
    same redshift can be applied to multiple spectra, or different redshifts can be
    applied to the same spectrum with appropriate input shapes.

    Parameters
    ----------
    z: float or numpy.ndarray
        The redshift(s) to apply, which can either be a single numerical value >= 0
        or else a numpy array of values >= 0.
    to_rest_frame: bool
        Divide wavelengths by 1+z instead of multiplying by 1+z.
    data_in: numpy.ndarray
        Structured numpy array containing input spectrum data to transform.
    data_out: numpy.ndarray
        Structured numpy array where output spectrum data should be written.
    wavelength: string or numpy array
        Either the field name in data_in containing wavelengths or else a numpy
        array of wavelength values.
    flux: string or numpy.ndarray
        Either the field name in data_in containing fluxes or else a numpy
        array of flux values.

    Returns
    -------
    result: numpy.ndarray
        Array of spectrum data with the redshift applied. Equal to data_out when
        set, otherwise a new array is allocated.
    """

    if not isinstance(z, np.ndarray):
        z = np.float(z)
    if np.any(z < 0):
        raise ValueError('Found invalid z < 0.')

    if data_in is not None or data_out is not None:
        if not isinstance(wavelength, basestring):
            raise ValueError('Invalid wavelength field name: {0}.'.format(wavelength))
        if not isinstance(flux, basestring):
            raise ValueError('Invalid flux field name: {0}.'.format(flux))

    if data_in is None:
        wavelength_in = wavelength
        wavelength = 'wavelength'
        flux_in = flux
        flux = 'flux'
    else:
        wavelength_in = data_in[wavelength]
        flux_in = data_in[flux]
    if wavelength_in.shape != flux_in.shape:
        raise ValueError('Input wavelength and flux arrays have different shapes: {0}, {1}'
            .format(wavelength_in.shape, flux_in.shape))

    if data_out is None:
        if data_in is None:
            dtype = [(wavelength, wavelength_in.dtype), (flux, flux_in.dtype)]
            data_out = np.empty(shape=wavelength_in.shape, dtype=dtype)
        elif isinstance(z, np.ndarray):
            data_out = np.empty(shape=np.broadcast(data_in, z).shape, dtype=data_in.dtype)
            data_out[...] = data_in
        else:
            data_out = np.copy(data_in)

    wavelength_out = data_out[wavelength]
    flux_out = data_out[flux]

    if to_rest_frame:
        zfactor = 1 / (1 + z)
    else:
        zfactor = 1 + z

    wavelength_out[:] = wavelength_in * zfactor
    flux_out[:] = flux_in * zfactor

    return data_out
