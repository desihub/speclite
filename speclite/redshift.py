import numpy as np


def transform(z, to_rest_frame=False, data_in=None, data_out=None,
              wavelength='wavelength', flux='flux'):
    """
    Apply a redshift to a spectrum.

    Parameters
    ----------
    z: float
        The redshift to apply. Must be >= 0.
    to_rest_frame: bool
        Divide wavelength and flux by 1+z instead of multiplying by 1+z.

    Returns
    -------
    result: numpy.ndarray
        Array of spectral data with the redshift applied.
    """

    if not isinstance(z, np.ndarray):
        z = np.array([z], dtype=float)
    if np.any(z < 0):
        raise ValueError('Found invalid z < 0.')

    if data_in is not None and isinstance(wavelength, basestring):
        wavelength_name = wavelength
        wavelength = data_in[wavelength_name]
    if data_in is not None and isinstance(flux, basestring):
        flux_name = flux
        flux = data_in[flux_name]

    if wavelength.shape != flux.shape:
        raise ValueError('wavelength and flux arrays have different shapes: {0}, {1}'
            .format(wavelength.shape, flux.shape))

    if data_out is None:
        if data_in is None:
            dtype = [('wavelength', wavelength.dtype), ('flux', flux.dtype)]
            data_out = np.empty(shape=wavelength.shape, dtype=dtype)
            wavelength_name = 'wavelength'
            flux_name = 'flux'
        else:
            data_out = np.copy(data_in)
    wavelength_out = data_out[wavelength_name]
    flux_out = data_out[flux_name]

    if to_rest_frame:
        zfactor = 1 / (1 + z)
    else:
        zfactor = 1 + z

    wavelength_out[:] = wavelength * zfactor
    flux_out[:] = flux * zfactor

    return data_out
