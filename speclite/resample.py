# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.ma as ma
import scipy.interpolate


def resample(data_in, x_in, x_out, y, data_out=None, kind='linear'):
    """
    Resample the data of one spectrum using interpolation.

    Dependent variables y1, y2, ... in the input data are resampled in the
    independent variable x using interpolation models y1(x), y2(x), ...
    evaluated on a new grid of x values. The independent variable will
    typically be a wavelength or frequency and the independent variables can
    be fluxes, inverse variances, etc.  For example:

    >>> data = np.ones((5,),
    ... [('wlen', float), ('flux', float), ('ivar', float)])
    >>> data['wlen'] = np.arange(4000, 5000, 200)
    >>> wlen_out = np.arange(4100, 4700, 200)
    >>> resample(data, 'wlen', wlen_out, ('flux', 'ivar'))
    array([(4100, 1.0, 1.0), (4300, 1.0, 1.0), (4500, 1.0, 1.0)],
          dtype=[('wlen', '<i8'), ('flux', '<f8'), ('ivar', '<f8')])

    The input grid can also be external to the structured array of spectral
    data, for example:

    >>> data = np.ones((5,), [('flux', float), ('ivar', float)])
    >>> wlen_in = np.arange(4000, 5000, 200)
    >>> wlen_out = np.arange(4100, 4900, 200)
    >>> resample(data, wlen_in, wlen_out, ('flux', 'ivar'))
    array([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
          dtype=[('flux', '<f8'), ('ivar', '<f8')])

    """
    if not isinstance(data_in, np.ndarray):
        raise ValueError('Invalid data_in type: {0}.'.format(type(data_in)))
    if data_in.dtype.fields is None:
        raise ValueError('Input data_in is not a structured array.')
    if len(data_in.shape) > 1:
        raise ValueError('Input data_in is multidimensional.')

    if isinstance(x_in, basestring):
        if x_in not in data_in.dtype.names:
            raise ValueError('No such x_in field: {0}.'.format(x_in))
        x_out_name = x_in
        x_in = data_in[x_in]
    else:
        if not isinstance(x_in, np.ndarray):
            raise ValueError('Invalid x_in type: {0}.'.format(type(x_in)))
        if x_in.shape != data_in.shape:
            raise ValueError('Incompatible shapes for x_in and data_in.')
        x_out_name = None
    x_type = np.promote_types(x_in.dtype, x_out.dtype)

    if ma.isMA(x_in) and np.any(x_in.mask):
        raise ValueError('Cannot resample masked x_in.')

    if not isinstance(x_out, np.ndarray):
        raise ValueError('Invalid x_out type: {0}.'.format(type(data_out)))

    dtype_out = []
    if x_out_name is not None:
        dtype_out.append((x_out_name, x_out.dtype))

    if isinstance(y, basestring):
        # Use a list instead of a tuple here so y_names can be used
        # to index data_in below.
        y_names = [y,]
    else:
        try:
            y_names = [name for name in y]
        except TypeError:
            raise ValueError('Invalid y type: {0}.'.format(type(y)))
    for not_first, y in enumerate(y_names):
        if y not in data_in.dtype.names:
            raise ValueError('No such y field: {0}.'.format(y))
        if not_first:
            if data_in[y].dtype != y_type:
                raise ValueError('All y fields must have the same type.')
        else:
            y_type = data_in[y].dtype
        dtype_out.append((y, y_type))

    y_shape = (len(y_names),)
    if ma.isMA(data_in):
        # Copy the structured 1D array into a 2D unstructured array
        # and set masked values to NaN.
        y_in = np.zeros(data_in.shape + y_shape, y_type)
        for i,y in enumerate(y_names):
            y_in[:,i] = data_in[y].filled(np.nan)
    else:
        y_in = data_in[y_names]
        # View the structured 1D array as a 2D unstructured array (without
        # copying any memory).
        y_in = y_in.view(y_type).reshape(data_in.shape + y_shape)
    # interp1d will only propagate NaNs correctly for certain values of `kind`.
    if np.any(np.isnan(y_in)):
        if kind not in ('nearest', 'linear', 'slinear', 0, 1):
            raise ValueError(
                'Interpolation kind not supported for masked data: {0}.'
                .format(kind))
    try:
        interpolator = scipy.interpolate.interp1d(
            x_in, y_in, kind=kind, axis=0, copy=False,
            bounds_error=False, fill_value=np.nan)
    except NotImplementedError:
        raise ValueError('Interpolation kind not supported: {0}.'.format(kind))

    shape_out = (len(x_out),)
    if data_out is None:
        data_out = np.empty(shape_out, dtype_out)
    else:
        if data_out.shape != shape_out:
            raise ValueError(
                'data_out has wrong shape: {0}. Expected: {1}.'
                .format(data_out.shape, shape_out))
        if data_out.dtype != dtype_out:
            raise ValueError(
                'data_out has wrong dtype: {0}. Expected: {1}.'
                .format(data_out.dtype, dtype_out))

    if x_out_name is not None:
        data_out[x_out_name][:] = x_out
    y_out = interpolator(x_out)
    for i,y in enumerate(y_names):
        data_out[y][:] = y_out[:,i]
    
    if ma.isMA(data_in) or np.any(np.isnan(y_out)):
        data_out = ma.MaskedArray(data_out)
        data_out.mask = False
        for y in y_names:
            data_out[y].mask = np.isnan(data_out[y].data)

    return data_out
