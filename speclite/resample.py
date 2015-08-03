# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.ma as ma
import scipy.interpolate


def resample(data_in, x_in, x_out, y, data_out=None, kind='linear'):
    """
    Resample spectral data...

    Builds an interpolated model y(x) from the input data sampled at ``x_in``
    and uses it to evaluate output data sampled at ``x_out``.
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

    if ma.isMA(data_in):
        # Replace masked entries with NaN.
        y_in = ma.filled(data_in[y_names], fill_value=np.nan)
    else:
        y_in = data_in[y_names]
    # View the structured 1D array as a 2D unstructured array (without
    # copying any memory).
    y_shape = (len(y_names),)
    y_in = y_in.view(y_type).reshape(data_in.shape + y_shape)
    # interp1d will only propagate NaNs correctly for certain values of kind.
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

    return data_out
