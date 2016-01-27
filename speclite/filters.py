# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Support for reading and applying filter response curves.

See :doc:`/filters` for more information.

Attributes
----------
default_wavelength_unit : :class:`astropy.units.Unit`
    The default wavelength units assumed when units are not specified.
    The same units are used to store wavelength values in internal arrays.
"""

import os
import os.path
import glob

import numpy as np

import scipy.interpolate
import scipy.integrate

import astropy.table
import astropy.units


default_wavelength_unit = astropy.units.Angstrom


def validate_wavelength_array(wavelength, min_length=0):
    """Validate a wavelength array for filter operations.

    Parameters
    ----------
    wavelength : array
        A 1D array of strictly increasing wavelength values with optional
        units.  If units are included, they must be convertible to
        :attr:`default_wavelength_unit`.  Otherwise, the
        :attr:`default_wavelength_unit` is assumed.
    min_length : int
        The minimum required length of the wavelength array.

    Raises
    ------
    ValueError
        Wavelength array is not 1D, or not strictly increasing, or below
        the minimum length.
    astropy.units.UnitConversionError
        The wavelength array has units that are not convertible to
        :attr:`default_wavelength_unit`
    """
    wavelength = np.asanyarray(wavelength)
    if len(wavelength.shape) != 1:
        raise ValueError('Wavelength array must be 1D.')
    if len(wavelength) < min_length:
        raise ValueError('Minimum length is {}.'.format(min_length))
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError('Wavelength values must be strictly increasing.')
    try:
        if wavelength.unit != default_wavelength_unit:
            # Try to convert to the default units. This will raise a UnitConversionError
            # if the current units are not convertible to the default units.
            wavelength = wavelength.to(default_wavelength_unit)
    except AttributeError:
        # No units present, so apply the default units.
        wavelength = wavelength * default_wavelength_unit
    return wavelength


class FilterResponse(object):
    """A filter response curve tabulated in wavelength.

    Some standard filters are included in this package and can be initialized
    using :func:`load`.  For example:

    >>> rband = load_filter('sdss2010-r')

    Objects behave like functions that evaluate their response at aribtrary
    wavelengths.  Wavelength units can be specified, or else default to
    :attr:`default_wavelength_unit`:

    >>> round(rband(6000 * astropy.units.Angstrom), 4)
    0.5323
    >>> round(rband(6000), 4)
    0.5323
    >>> round(rband(0.6 * astropy.units.micron), 4)
    0.5323

    Filters can be also evaluated for an arbitrary array of wavelengths,
    returning a numpy array of response values:

    >>> np.round(rband([5980, 6000, 6020]), 4)
    array([ 0.5309,  0.5323,  0.5336])

    Parameters
    ----------
    wavelength : array
        A :func:`valid array <validate_wavelength_array>` of wavelengths.
    response : array
        A dimensionless 1D array of filter response values corresponding to
        each wavelength.  Response values must be non-negative and cannot all
        be zero. The bounding response values must be zero, and the response
        is assumed to be zero outside of the specified wavelength range.
    meta : dict
        A dictionary of metadata which must include values for the keys listed
        :doc:`here </filters>`.  Additional keys are also permitted.

    Attributes
    ----------
    wavelength : numpy.ndarray
        Numpy array of wavelength values passed to our constructor, after
        trimming any extra leading or trailing wavelengths with zero response.
    response : numpy.ndarray
        Numpy array of response values passed to our constructor, after
        trimming any extra leading or trailing zero response values.
    interpolator : :class:`scipy.interpolate.interp1d`
        Linear interpolator of our response function that returns zero for
        all values outside our wavelength range.  Should normally be evaluated
        through our :meth:`__call__` convenience method.

    Raises
    ------
    ValueError
        Invalid wavelength or response input arrays, or missing required keys
        in the input metadata.
    """
    def __init__(self, wavelength, response, meta):

        self.wavelength = validate_wavelength_array(wavelength, min_length=3)
        self.response = np.asanyarray(response)
        if len(self.wavelength) != len(self.response):
            raise ValueError('Arrays must have same length.')

        try:
            if self.response.decompose().unit != u.dimensionless_unscaled:
                raise ValueError('Response must be dimensionless.')
            # Convert response values to a plain numpy array.
            self.response = self.response.value
        except AttributeError:
            # response has no units assigned, which is fine.
            pass

        # Check for a valid response curve.
        if np.any(self.response < 0):
            raise ValueError('Response values must be non-negative.')
        if np.all(self.response == 0):
            raise ValueError('Response values cannot all be zero.')
        if not self.response[0] == 0 and self.response[-1] == 0:
            raise ValueError('Response must go to zero on both sides.')

        # Trim any extra leading and trailing zeros.
        non_zero = np.where(self.response > 0)[0]
        start, stop = non_zero[0] - 1, non_zero[-1] + 2
        if stop - start < len(self.wavelength):
            self.wavelength = self.wavelength[start: stop]
            self.response = self.response[start: stop]

        # Check for the required metadata fields.
        self.meta = dict(meta)
        for required in ('group_name', 'band_name'):
            if required not in self.meta:
                raise ValueError(
                    'Metadata missing required key "{}".'.format(required))

        # Create a linear interpolator of our response function that returns zero
        # outside of our wavelength range.
        self.interpolator = scipy.interpolate.interp1d(
            self.wavelength.value, self.response, kind='linear',
            copy=False, assume_sorted=True,
            bounds_error=False, fill_value=0.)


    def __call__(self, wavelength):
        """Evaluate the filter response at arbitrary wavelengths.

        Parameters
        ----------
        wavelength : array or float
            A single wavelength value or an array of wavelengths.
            If units are included, they will be correctly interpreted.
            Otherwise :attr:`default_wavelength_unit` is assumed.

        Returns
        -------
        numpy.ndarray
            Numpy array of response values corresponding to each input
            wavelength.

        Raises
        ------
        astropy.units.UnitConversionError
            Input wavelength(s) have unit that is not convertible to
            :attr:`default_wavelength_unit`.
        """
        # Use asanyarray() so that a Quantity with units is not copied here.
        wavelength = np.asanyarray(wavelength)
        try:
            wavelength = wavelength.to(default_wavelength_unit).value
        except AttributeError:
            # No units present, so assume the default units.
            pass
        response = self.interpolator(wavelength)
        # If the input was scalar, return a scalar.
        if response.shape == ():
            response = np.asscalar(response)
        return response


    def convolve_with_array(self, wavelength, values, axis=-1,
                            extrapolate=False, interpolate=False):
        """
        """
        convolution = FilterReponseConvolution(self, wavelength, extrapolate, interpolate)
        return convolution(values, axis=axis)


    def convolve_with_function(self, function):
        """
        """
        pass

    def get_effective_wavelength(self):
        pass


def load_filter(name):
    """Load a filter response by name.

    See :doc:`/filters` for details on the filter response file format and
    the available standard filters.

    Parameters
    ----------
    name : str
        Name of the filter response to load, which should have the format
        "<group_name>-<band_name>".

    Returns
    -------
    FilterResponse
        A :class:`FilterResponse` object for the requested filter.
    """
    file_name = astropy.utils.data.get_pkg_data_filename(
        'data/filters/{}.txt'.format(name))
    table = astropy.table.QTable.read(file_name, format='ascii.ecsv')
    return FilterResponse(table['wavelength'], table['response'], table.meta)
