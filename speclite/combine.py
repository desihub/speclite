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

    # Find the intersection of field names in both input datasets.
    shared_fields = set(data1_in.dtype.names) & set(data2_in.dtype.names)
    if len(shared_fields) == 0:
        raise ValueError('data1_in and data2_in have no fields in common.')

    if join is not None:
        if join not in shared_fields:
            raise ValueError('Invalid join field name: {0}.'.format(join))
        if not np.array_equal(data1_in[join], data2_in[join]):
            raise ValueError('Inputs do not have identical join field values.')

    try:
        add_names = iter(add)
    except TypeError:
        add_names = (add,)
    for name in add_names:
        if name not in shared_fields:
            raise ValueError('Invalid add field name: {0}.'.format(name))

    if weight is not None:
        if weight not in shared_fields:
            raise ValueError('Invalid weight field name: {0}.'.format(weight))

    return data_out
