import numpy as np
import numpy.ma as ma


def transform(z_in, z_out, data_in=None, data_out=None,
              wavelength='wavelength', flux='flux'):
    """
    Transform spectral data from redshift z_in to z_out.

    Each quantity X is transformed according to a power law::

        X_out = X_in * ((1 + z_out) / (1 + z_in))**exponent

    where all non-zero exponents are specified with the ``rules`` argument.
    The usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ apply so that, for
    example, the same redshift can be applied to multiple spectra, or different redshifts
    can be applied to the same spectrum with appropriate input shapes.

    Input arrays can have `units
    <http://astropy.readthedocs.org/en/latest/units/index.html>`__ but these will not be
    used or propagated to the output (since numpy structured arrays do not support
    per-column units).  Input arrays can have associated `masks
    <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ and these will be
    propagated to the output.

    Parameters
    ----------
    z_in: float or numpy.ndarray
        Redshift(s) of the input spectral data, which must all be >= 0.
    z_out: float or numpy.ndarray
        Redshift(s) of the output spectral data, which must all be >= 0.
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
        Array of spectrum data with the redshift transform applied. Equal to data_out
        when set, otherwise a new array is allocated. The array shape will be the result
        of broadcasting the input z and spectral data arrays.
    """

    if not isinstance(z_in, np.ndarray):
        z_in = np.float(z_in)
    if np.any(z_in < 0):
        raise ValueError('Found invalid z_in < 0.')
    if not isinstance(z_out, np.ndarray):
        z_out = np.float(z_out)
    if np.any(z_out < 0):
        raise ValueError('Found invalid z_out < 0.')
    z_factor = (1.0 + z_out) / (1.0 + z_in)

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
            shape = np.broadcast(wavelength_in, z_factor).shape
            dtype = [(wavelength, wavelength_in.dtype), (flux, flux_in.dtype)]
            if ma.isMA(wavelength_in) or ma.isMA(flux_in):
                data_out = ma.empty(shape, dtype=dtype)
                data_out.mask = False
                if ma.isMA(wavelength_in):
                    data_out[wavelength].mask[...] = wavelength_in.mask
                if ma.isMA(flux_in):
                    data_out[flux].mask[...] = flux_in.mask
            else:
                data_out = np.empty(shape=shape, dtype=dtype)
        else:
            shape = np.broadcast(data_in, z_factor).shape
            if ma.isMA(data_in):
                # The next line fails with shape=out_shape.
                # https://github.com/numpy/numpy/issues/6106
                data_out = ma.empty(shape, dtype=data_in.dtype)
            else:
                data_out = np.empty(shape=shape, dtype=data_in.dtype)
            data_out[...] = data_in
    else:
        shape = np.broadcast(data_in, z_factor).shape
        if data_out.shape != shape:
            raise ValueError('Invalid data_out shape {0}, expected {1}.'
                .format(data_out.shape, shape))

    wavelength_out = data_out[wavelength]
    flux_out = data_out[flux]

    wavelength_out[:] = wavelength_in * z_factor
    flux_out[:] = flux_in / z_factor

    return data_out
