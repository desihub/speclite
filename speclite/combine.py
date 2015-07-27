# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.ma as ma


def accumulate(data1_in, data2_in, coef1=1., coef2=1.,
               join=None, add=None, weight=None, data_out=None):
    """
    Combine the data from two spectra.

    Fields from both inputs with matching names can be combined in several
    ways:
    * join: use this field to match rows between the inputs on their last axis.
    * weight: use this field to weight any addition rules.
    * add: calculate (weighted) sum and sum of squares.
    * ufunc: apply an arbitrary numpy ufunc, e.g., logical and/or.
    At most one field can be joined and, if no field is joined, then both
    inputs must have the same shape and will be joined based on their indices.

    Fields that are unique to one input or the other are not propagated to the
    output unless specified by a 'copy' rule.  It is an error to copy a field
    that appears in both inputs.

    accumulate(data1, data2, rules=dict(join='wlen', add='flux', weight='ivar'))
    """
    if not isinstance(data1_in, np.ndarray):
        raise ValueError('data1_in is not a numpy array.')
    if not isinstance(data2_in, np.ndarray):
        raise ValueError('data2_in is not a numpy array.')
    if data_out is not None and not isinstance(data_out, np.ndarray):
        raise ValueError('data_out is not a numpy array.')

    if data1_in.shape != data2_in.shape:
        raise ValueError(
            'Inputs have different shapes: {0} != {1}.'
            .format(data1_in.shape, data2_in.shape))
    shape_out = data1_in.shape

    data1_fields = data1_in.dtype.fields
    if data1_fields is None:
        raise ValueError('Input data1_in is not a structured array.')
    data2_fields = data2_in.dtype.fields
    if data2_fields is None:
        raise ValueError('Input data2_in is not a structured array.')
    dtype_out = []

    # Find the intersection of field names in both input datasets.
    shared_fields = set(data1_fields.keys()) & set(data2_fields.keys())
    if len(shared_fields) == 0:
        raise ValueError('Inputs have no fields in common.')

    if join is None:
        join_names = ()
    elif isinstance(join, basestring):
        join_names = (join,)
    else:
        try:
            join_names = iter(join)
        except TypeError:
            raise ValueError('Invalid join type: {0}.'.format(type(join)))
    for name in join_names:
        if name not in shared_fields:
            raise ValueError('Invalid join field name: {0}.'.format(name))
        if not np.array_equal(data1_in[name], data2_in[name]):
            raise ValueError(
                'Cannot join on unmatched field: {0}.'.format(name))
        dtype1 = data1_fields[name][0]
        dtype2 = data1_fields[name][0]
        dtype_out.append((name, np.promote_types(dtype1, dtype2)))

    if add is None:
        add_names = ()
    elif isinstance(add, basestring):
        add_names = (add,)
    else:
        try:
            add_names = iter(add)
        except TypeError:
            raise ValueError('Invalid add type: {0}.'.format(type(add)))
    for name in add_names:
        if name not in shared_fields:
            raise ValueError('Invalid add field name: {0}.'.format(name))
        dtype1 = data1_fields[name][0]
        dtype2 = data1_fields[name][0]
        '''
        if not isinstance(dtype1, np.number):
            raise ValueError('Cannot add non-numeric type: {0}.'.format(dtype1))
        if not isinstance(dtype2, np.number):
            raise ValueError('Cannot add non-numeric type: {0}.'.format(dtype2))
        '''
        dtype_out.append((name, np.promote_types(dtype1, dtype2)))

    if weight is not None:
        if weight not in shared_fields:
            raise ValueError('Invalid weight field name: {0}.'.format(weight))
        weight1 = data1_in[weight]
        weight2 = data2_in[weight]
    else:
        weight1 = np.ones_like(shape_out)
        weight2 = np.ones_like(shape_out)

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

    for name in add_names:
        data_out[name][:] = (
            coef1*weight1*data1_in[name] + coef2*weight2*data2_in[name])

    return data_out
