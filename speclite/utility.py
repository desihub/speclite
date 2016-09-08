# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Apply redshift transformations to wavelength, flux, inverse variance, etc.
"""
from __future__ import print_function, division

import collections

import numpy as np
import numpy.ma
import astropy.table


def get_options(kwargs_in, **defaults):
    """Extract predefined options from keyword arguments.

    Parameters
    ----------
    kwargs_in : dict
        A dictionary of input keyword arguments to process.
    **defaults
        Arbitrary keyword arguments that specify the names of the predefined
        options and their default values.

    Returns
    -------
    tuple
        A tuple (kwargs_out, options) where kwargs_out is a copy of
        kwargs_in with any options deleted and options combines the
        defaults and input options.
    """
    kwargs_out = kwargs_in.copy()
    options = {}
    for name in defaults:
        options[name] = kwargs_in.get(name, defaults[name])
        if name in kwargs_out:
            del kwargs_out[name]
    return kwargs_out, options


def empty_like(array, shape=None, dtype=None, add_mask=False):
    """Create an array with unit and mask properties taken from a prototype.

    Supports numpy regular and masked arrays, as well as astropy Quantity,
    Column and MaskedColumn objects. In the case of Column and MaskedColumn
    objects, all associated metadata (name, description, etc) will also be
    propagated to the new object.

    Parameters
    ----------
    array : numpy or astropy array-like object
        An array whose properties will be propagated to the newly created
        array.  Specifically, any associated units will be copied and
        a mask will be included if this object has a mask.
    shape : tuple or None
        Shape of the returned array. Use the shape of array if None.
    dtype : numpy data type or None
        Data type of the return array. Use the data type of array if None.
    add_mask : bool
        Include a mask in the return array even if the prototype array
        does not have a mask.

    Returns
    -------
    numpy or astropy array-like object
        Object will have the specified shape and dtype, but have its other
        properties copied from the input array.  The returned array data
        will not be initialized.
    """
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype

    if numpy.ma.isMaskedArray(array) or add_mask:
        # The keyword version fails in older versions of numpy.ma
        # https://github.com/numpy/numpy/issues/6106
        #result = numpy.ma.empty(shape=shape, dtype=dtype)
        result = numpy.ma.empty(shape, dtype)
    else:
        result = np.empty(shape=shape, dtype=dtype)

    if isinstance(array, astropy.table.Column):
        # Wrap the new array in a Column or MaskedColumn.
        result = array.copy(data=result, copy_data=False)
    else:
        try:
            # Add any units to create a Quantity.
            result = result * array.unit
        except AttributeError:
            pass

    return result


def tabular_like(tabular, columns, dimension=1):
    """
    Since columns is a dict, the order of columns in the created table
    is arbitrary. Used collections.OrderedDict to specify the order.
    """
    # Check for consistent shapes of the columns and determine the
    # shape of the rows for the new tabular object.
    for i, name in enumerate(columns.keys()):
        shape = columns[name].shape
        if i and shape[:dimension] != rows_shape:
            raise ValueError('Column {0} has invalid shape {1}.'
                             .format(name, shape))
        else:
            rows_shape = shape[:dimension]

    if isinstance(tabular, astropy.table.Table):
        # Astropy table.
        if dimension != 1:
            raise ValueError('Row shape must be 1D for astropy table.')
        return tabular.__class__(columns.values(), names=columns.keys())

    if hasattr(tabular, 'fields'):
        # Numpy structured array.
        dtype = []
        for name in columns.keys():
            shape = columns[name].shape
            if len(shape) > dimension:
                dtype.append((name, columns[name].dtype, shape[dimension:]))
            else:
                dtype.append((name, columns[name].dtype))
        if numpy.ma.isMaskedArray(tabular):
            result = numpy.ma.array(rows_shape, dtype)
        else:
            result = numpy.array(rows_shape, dtype)
        # Copy the column data into the newly created structured array.
        for name in columns.keys():
            result[name] = columns[name]
        return result

    # If we get here, this is an unsupported tabular type.
    raise ValueError('Unsupported tabular type {0}.'.format(type(tabular)))


def prepare_data(mode, args, kwargs):
    """Prepare data for an algorithm.

    Parameters
    ----------
    mode : tuple or str
        Either the desired shape of each array or else one of the strings
        'in_place' or 'read_only'.  When a shape tuple is specified, the
        returned arrays will be newly created with the specified shape.
        With 'in_place', the input arrays will be returned if possible or
        this method will raise a ValueError (for example, if one of the
        input arrays is not already an instance of a numpy array). With
        'read_only', read-only views of the input arrays will be returned,
        using temporary copies when necessary (for example, if one of the
        input arrays is not already an instance of a numpy array).
    args : tuple
        Non-keyword arguments that define the data to prepare. This should
        either be empty or contain a single tabular object that is
        internally organized into named arrays.  Instances of a numpy
        structured array, possibly including astropy units or a mask,
        and instances of an astropy table are both supported.
    kwargs : dict
        Keyword arguments the define the data to prepare. This option can
        not be combined with a non-empty args. When only keyword arguments
        are present, each one is assumed to name a value that is convertible
        to a numpy unstructured array of the same shape.

    Returns
    -------
    tuple
        A tuple (arrays, value) where arrays is a dictionary of array
        names and corresponding numpy unstructured arrays and result is an
        appropriate return value for a function that updates these arrays:
        a tabular type, numpy structured array or the arrays dictionary.
        The original column order will be preserved for an input tabular
        object by returning a collection.OrderedDict for arrays.
    """
    # Check for a valid mode.
    if isinstance(mode, tuple):
        output_shape = mode
    elif mode in ('in_place', 'read_only'):
        output_shape = None
    else:
        raise ValueError('Invalid mode {0}.'.format(mode))

    # Check for valid args and kwargs.
    if len(args) > 1:
        raise ValueError(
            'Must provide zero or one non-keyword args.')
    elif len(args) == 1 and len(kwargs) > 0:
        raise ValueError(
            'Cannot provide both keyword and non-keyword args.')

    # A single non-keyword dictionary arg is converted into kwargs.
    if len(args) == 1 and isinstance(args[0], dict):
        kwargs = args[0]
        del args[0]

    if len(args) == 1:
        # The unique non-keyword argument must be a tabular type.
        tabular = args[0]
        # Test for an astropy.table.Table.  We check for the colnames
        # attribute instead of using isinstance() so that this test fails
        # gracefully if astropy is not installed.
        if hasattr(tabular, 'colnames'):
            if output_shape is not None:
                # At this point, we need astropy to create new columns.
                import astropy.table
                # Make a copy of this table but not its underlying data.
                tabular = tabular.copy(copy_data=False)
                # Replace each column with a new column of the new shape.
                for name in tabular.colnames:
                    c = tabular[name]
                    new_column = astropy.table.Column(
                        name=name, description=c.description,
                        unit=c.unit, meta=c.meta,
                        data=np.empty(output_shape, dtype=c.dtype))
                    try:
                        tabular.replace_column(name, new_column)
                    except AttributeError:
                        # Older versions of astropy (including the LTS
                        # version 1.0) do not implement this method, so
                        # we copy its implementation here for now.
                        t = tabular.__class__([new_column], names=[name])
                        new_cols = collections.OrderedDict(tabular.columns)
                        new_cols[name] = t[name]
                        tabular._init_from_cols(new_cols.values())
            # Preserve the order of columns in the input table.
            arrays = collections.OrderedDict()
            for name in tabular.colnames:
                arrays[name] = tabular[name]

        # Test for a numpy structured array.
        elif (hasattr(tabular, 'dtype') and
              getattr(tabular.dtype, 'names', None) is not None):
            if output_shape is not None:
                # Make a copy of this structured array.
                tabular = tabular.copy()
                # Change the shape of the new copy.
                if hasattr(tabular, 'mask'):
                    tabular = numpy.ma.resize(tabular, output_shape)
                else:
                    tabular.resize(output_shape)
            # Preserve the order of columns in the input structured array.
            arrays = collections.OrderedDict()
            for name in tabular.dtype.names:
                arrays[name] = tabular[name]
        else:
            raise ValueError(
                'Cannot prepare input from tabular type {0}.'
                .format(type(tabular)))
        # The tabular object is our value.
        value = tabular

    else:
        # Each value must be convertible to an unstructured numpy array.
        arrays = {}
        for name in kwargs:
            if output_shape is not None:
                arrays[name] = np.asanyarray(kwargs[name]).copy()
                if hasattr(arrays[name], 'mask'):
                    arrays[name] = numpy.ma.resize(arrays[name], output_shape)
                else:
                    arrays[name].resize(output_shape)
            elif mode == 'in_place':
                # This will fail unless the array is already an instance
                # of a numpy array.
                try:
                    arrays[name] = kwargs[name].view()
                except AttributeError:
                    raise ValueError('Cannot update array "{0}" in place.'
                                     .format(name))
            else:
                # Create a view with no memory copy when possible.
                arrays[name] = np.asanyarray(kwargs[name])
            # Check that this is not a structured array.
            if arrays[name].dtype.names is not None:
                raise ValueError(
                    'Cannot pass structured array "{0}" as keyword arg.'
                    .format(name))
        # The dictionary of arrays is our value.
        value = arrays

    # Verify that all values are subclasses of np.ndarray
    # and create read-only views if requested.
    for i, name in enumerate(arrays):
        assert isinstance(arrays[name], np.ndarray)
        if arrays[name].flags.writeable and mode == 'read_only':
            arrays[name] = arrays[name].view()
            arrays[name].flags.writeable = False

    return arrays, value
