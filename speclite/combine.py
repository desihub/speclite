# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.ma as ma


def accumulate(data1_in, data2_in, data_out=None,
               join=None, add=None, weight=None):
    """
    Combine the data from two spectra.

    Values x1 and x2 with corresponding weights w1 and w2 are combined as::

        x12 = (w1*x1 + w2*x2)/(w1 + w2)

    If no weight field is present for either input, a weight of one is used.
    If either input array is `masked
    <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__, weights
    for masked entries will be set to zero. The output contains values for
    x12 and the accumulated weight:

        w12 = w1 + w2

    For example::

    >>> data1 = np.ones((10,), dtype=[('flux', float), ('ivar', float)])
    >>> data2 = np.ones((10,), dtype=[('flux', float), ('ivar', float)])
    >>> result = accumulate(data1, data2, add='flux', weight='ivar')
    >>> result[:3]
    array([(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)],
          dtype=[('flux', '<f8'), ('ivar', '<f8')])

    Any fields common to both inputs can also be copied to the output::

    >>> data1 = np.ones((10,),
    ... dtype=[('wlen', float), ('flux', float)])
    >>> data2 = np.ones((10,),
    ... dtype=[('wlen', float), ('flux', float)])
    >>> result = accumulate(data1, data2, join='wlen', add='flux')
    >>> result[:3]
    array([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
          dtype=[('wlen', '<f8'), ('flux', '<f8')])

    The actual calculation of x12 uses the expression::

        x12 = x1 + (x2 - x1)*w2/(w1 + w2)

    which has `better numerical properties
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #Weighted_incremental_algorithm>`__ when many spectra are
    iteratively accumulated using the following pattern::

    >>> result = None
    >>> data = np.ones((10,100),
    ... dtype=[('wlen', float), ('flux', float), ('ivar', float)])
    >>> for row in data:
    ...     result = accumulate(data1_in=result, data2_in=row, data_out=result,
    ...                         join='wlen', add='flux', weight='ivar')
    >>> result[:3]
    array([(1.0, 1.0, 10.0), (1.0, 1.0, 10.0), (1.0, 1.0, 10.0)],
          dtype=[('wlen', '<f8'), ('flux', '<f8'), ('ivar', '<f8')])

    With this pattern, the result array is allocated on the first iteration
    and then re-used for all subsequent iterations.
    and join fields are copied
    on the first iteration

    Parameters
    ----------
    data1_in: numpy.ndarray
        First structured numpy array of input spectral data. Can be None.
    data2_in: numpy.ndarray
        Second structured numpy array of input spectral data.
    data_out: numpy.ndarray
        Structured numpy array where output spectrum data should be written. If
        None is specified, then an appropriately sized array will be allocated
        and returned. Use this method to take control of the memory allocation
        and, for example, re-use the same output array for iterative
        accumulation.
    join: string or iterable or None.
        A field name or a list of field names that are present in both inputs
        with identical values, and should be included in the output.
    add: string or iterable or None.
        A field name or a list of field names that are present in both inputs
        and whose values, x1 and x2, should be accumulated as x12 in the
        output.
    weight: string or None.
        The name of a field whose values provide the weights w1 and w2 used
        to calculate the accumulated x12 = w1*x1 + w2*x2.  If the named field
        is not present in either input a weight value of one will be used.
        The output array will contain a field with this name, if it is not
        None, containing values for w12.
    """
    if data1_in is not None and not isinstance(data1_in, np.ndarray):
        raise ValueError('data1_in is not a numpy array.')
    if not isinstance(data2_in, np.ndarray):
        raise ValueError('data2_in is not a numpy array.')
    if data_out is not None and not isinstance(data_out, np.ndarray):
        raise ValueError('data_out is not a numpy array.')

    if data1_in is not None:
        if data1_in.shape != data2_in.shape:
            raise ValueError(
                'Inputs have different shapes: {0} != {1}.'
                .format(data1_in.shape, data2_in.shape))
        data1_fields = data1_in.dtype.fields
        if data1_fields is None:
            raise ValueError('Input data1_in is not a structured array.')

    data2_fields = data2_in.dtype.fields
    if data2_fields is None:
        raise ValueError('Input data2_in is not a structured array.')
    shape_out = data2_in.shape
    dtype_out = []

    # Find the intersection of field names in both input datasets.
    if data1_in is not None:
        shared_fields = set(data1_fields.keys()) & set(data2_fields.keys())
        if len(shared_fields) == 0:
            raise ValueError('Inputs have no fields in common.')
    else:
        shared_fields = set(data2_fields.keys())

    def prepare_names(arg, label):
        if arg is None:
            names = ()
        elif isinstance(arg, basestring):
            names = (arg,)
        else:
            try:
                names = iter(arg)
            except TypeError:
                raise ValueError(
                    'Invalid {0} type: {1}.'.format(label, type(arg)))
        for name in names:
            if name not in shared_fields:
                raise ValueError(
                    'Invalid {0} field name: {1}.'.format(label, name))
            if data1_in is not None:
                dtype1 = data1_fields[name][0]
                dtype2 = data2_fields[name][0]
                dtype_out.append((name, np.promote_types(dtype1, dtype2)))
            else:
                dtype_out.append((name, data2_fields[name][0]))
        return names

    join_names = prepare_names(join, 'join')
    add_names = prepare_names(add, 'add')

    if data1_in is not None:
        for name in join_names:
            if not np.array_equal(data1_in[name], data2_in[name]):
                raise ValueError(
                    'Cannot join on unmatched field: {0}.'.format(name))

    if weight is not None:
        if not isinstance(weight, basestring):
            raise ValueError('Invalid weight type: {0}.'.format(type(weight)))
        if data1_in is not None:
            if weight in data1_fields:
                weight1 = data1_in[weight]
            else:
                weight1 = np.ones(shape_out)
        if weight in data2_fields:
            weight2 = data2_in[weight]
        else:
            weight2 = np.ones(shape_out)
        if data1_in is not None:
            dtype_out.append(
                (weight, np.promote_types(weight1.dtype, weight2.dtype)))
        else:
            dtype_out.append((weight, weight2.dtype))
    else:
        if data1_in is not None:
            weight1 = np.ones(shape_out)
        weight2 = np.ones(shape_out)

    # Set weights to zero for any masked elements. In the (unlikely?) case
    # that different fields have different masks, we use the logical OR of
    # all named join/add/weight fields.
    if data1_in is not None and ma.isMA(data1_in):
        mask = np.zeros(shape_out, dtype=bool)
        for name in join_names:
            mask = mask | data1_in[name].mask
        for name in add_names:
            mask = mask | data1_in[name].mask
        if weight is not None:
            mask = mask | data1_in[weight].mask
        weight1[mask] = 0
    if ma.isMA(data2_in):
        mask = np.zeros(shape_out, dtype=bool)
        for name in join_names:
            mask = mask | data2_in[name].mask
        for name in add_names:
            mask = mask | data2_in[name].mask
        if weight is not None:
            mask = mask | data2_in[weight].mask
        weight2[mask] = 0

    if len(dtype_out) == 0:
        raise ValueError('No result fields specified.')

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

    # We do not need to copy join fields if data_out uses the same memory
    # as one of our input arrays.
    if data_out.base is None or data_out.base not in (data1_in, data2_in):
        for name in join_names:
            data_out[name][:] = data2_in[name]

    if data1_in is None:
        for name in add_names:
            data_out[name][:] = weight2*data2_in[name]
        if weight is not None:
            data_out[weight][:] = weight2
    else:
        # Accumulate add fields.
        weight_sum = weight1 + weight2
        for name in add_names:
            data_out[name][:] = (
                data1_in[name] +
                (data1_in[name] - data2_in[name])*weight2/weight_sum)
        if weight is not None:
            data_out[weight][:] = weight_sum

    return data_out
