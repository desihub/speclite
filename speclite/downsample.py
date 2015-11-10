# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.ma as ma


def downsample(data_in, downsampling, weight=None, axis=-1, start_index=0,
               auto_trim=True, data_out=None):
    """Downsample spectral data by a constant factor.

    Downsampling consists of dividing the input data into fixed-size groups of
    consecutive bins, then calculated downsampled values as weighted averages
    within each group.  The basic usage is:

    >>> data = np.ones((6,), dtype=[('flux', float), ('ivar', float)])
    >>> downsample(data, downsampling=2, weight='ivar')
    array([(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)],
          dtype=[('flux', '<f8'), ('ivar', '<f8')])

    Parameters
    ----------
    data_in : numpy.ndarray or numpy.ma.MaskedArray
        Structured numpy array containing input spectrum data to downsample.
    downsampling : int
        Number of consecutive bins to combine into each downsampled bin.
        Must be at least one and not larger than the input data size.
    weight : string or None.
        The name of a field whose values provide the weights to use for
        downsampling.  When None, a weight value of one will be used.
        The output array will contain a field with this name, unless it is
        None, containing values of the downsampled weights.  All weights must
        be non-negative.
    start_index : int
        Index of the first bin to use for downsampling. Any bins preceeding
        the start bin will not be included in the downsampled results. Negative
        indices are not allowed.
    axis : int
        Index of the axis to perform downsampling in.
    auto_trim : bool
        When True, any bins at the end of the input data that do not fill a
        complete downsampled bin will be automatically (and silently) trimmed.
        When False, a ValueError will be raised.
    data_out : numpy.ndarray
        Structured numpy array where output spectrum data should be written. If
        none is specified, then an appropriately sized array will be allocated
        and returned. Use this method to take control of the memory allocation
        and, for example, re-use the same output array for a sequence of
        downsampling operations.

    Returns
    -------
    numpy.ndarray or numpy.ma.MaskedArray
        ...
    """
    if not isinstance(data_in, np.ndarray):
        raise ValueError('Invalid data_in type: {0}.'.format(type(data_in)))
    if data_out is not None and not isinstance(data_out, np.ndarray):
        raise ValueError('Invalid data_out type: {0}.'.format(type(data_out)))

    shape_in = data_in.shape
    try:
        num_bins = shape_in[axis]
    except IndexError:
        raise ValueError('Invalid axis = {}.'.format(axis))

    if downsampling < 1 or downsampling >= num_bins:
        raise ValueError('Invalid downsampling = {}.'.format(downsampling))
    if start_index < 0 or start_index + downsampling > num_bins:
        raise ValueError('Invalid start_index = {}.'.format(start_index))

    num_downsampled = (num_bins - start_index) // downsampling
    if num_downsampled <= 0:
        raise ValueError('Incompatible downsampling = {} and start_index = {}.'
                         .format(downsampling, start_index))
    stop_index = start_index + num_downsampled * downsampling
    assert stop_index <= num_bins
    if stop_index < num_bins and not auto_trim:
        raise ValueError(
            'Input data does not evenly divide with downsampling = {}.'
            .format(downsampling))

    if weight is not None:
        if not isinstance(weight, basestring):
            raise ValueError('Invalid weight type: {0}.'.format(type(weight)))
        if weight in data_in.dtype.fields:
            weights_in = data_in[weight]
            if np.any(weights_in < 0):
                raise ValueError('Some input weights < 0.')
        else:
            raise ValueError('No such weight field: {}.'.format(weight))
    else:
        weights_in = np.ones(shape_in)

    shape_out = list(shape_in)
    shape_out[axis] = num_downsampled
    expanded_shape = list(shape_in)
    expanded_shape[axis] = downsampling
    expanded_shape.insert(axis, num_downsampled)
    sum_axis = axis + 1 if axis >= 0 else len(shape_in) + axis + 1

    dtype_out = data_in.dtype
    if data_out is None:
        data_out = np.empty(shape=shape_out, dtype=data_in.dtype)
    else:
        if data_out.shape != shape_out:
            raise ValueError(
                'data_out has wrong shape: {0}. Expected: {1}.'
                .format(data_out.shape, shape_out))
        if data_out.dtype != dtype_out:
            raise ValueError(
                'data_out has wrong dtype: {0}. Expected: {1}.'
                .format(data_out.dtype, dtype_out))

    # Loop over fields in the input data.
    weights_out = np.sum(
        weights_in[start_index:stop_index].reshape(expanded_shape),
        axis=sum_axis)
    for field in data_in.dtype.fields:
        if field == weight:
            continue
        weighted = (
            weights_in[start_index:stop_index] *
            data_in[field][start_index:stop_index])
        data_out[field] = np.sum(
            weighted.reshape(expanded_shape), axis=sum_axis) / weights_out
    if weight is not None:
        data_out[weight] = weights_out

    return data_out
