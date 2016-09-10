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


def get_tabular_type(tabular):
    """Determine the type of a tabular object.

    This function defines the set of supported tabular types and how they
    are identified.

    Parameters
    ----------
    tabular : numpy or astropy tabular object

    Returns
    -------
    'table', 'numpy', 'dict' or None
        A string indicating which type of supported tabular object this is.
    """
    if isinstance(tabular, astropy.table.Table):
        return 'table'
    elif hasattr(tabular, 'dtype') and hasattr(tabular.dtype, 'fields'):
        return 'numpy'
    elif isinstance(tabular, dict):
        return 'dict'
    else:
        return None


def tabular_like(tabular, columns, dimension=0):
    """Create a tabular object from column data.

    This function is normally used together with :func:`prepare_data` in
    order to implement algorithms on tabular data that operate only on
    numpy arrays and are insulated from the details of different tabular
    objects.

    The supported tabular objects are numpy structured arrays and masked arrays,
    astropy tables and masked tables, and dictionaries of array-like objects.

    Parameters
    ----------
    tabular : numpy or astropy tabular object
        A tabular object to use as a prototype for the newly created object.
    columns : dict or OrderedDict
        A dictionary array-like objects indexed by column name. Array-like
        objects must be compatible with the tabular object being created.
        Use an OrderedDict to control the order of columns in the newly
        created table.
    dimension : int
        The dimension of the created tabular object.  A table is normally
        indexed by rows, so is 1D, but more complex partitioning is possible
        with numpy structured arrays.  Each column array must have the same
        value of shape[:dimension].  When the value is zero, it will be
        automatically calculated to be either 1 for an astropy table,
        as large as possible for a numpy structured array, or zero for
        a dictionary of arrays.

    Returns
    -------
    numpy or astropy tabular object
        The returned type will match the input tabular object.
    """
    ttype = get_tabular_type(tabular)
    if ttype is None:
        raise ValueError('Unsupported tabular type {0}.'.format(type(tabular)))

    # Extract a list of shapes for each array.
    shapes = [array.shape for array in columns.values()]

    # Calculate the dimension automatically if requested.
    if dimension == 0:
        if ttype == 'table':
            # Tables must have dimension of one.
            dimension = 1
            if not np.all([
                shape[0] == shapes[0][0] for shape in shapes[1:]]):
                raise ValueError('Column arrays have inconsistent shapes.')
        elif ttype == 'numpy':
            # Find the largest possible dimensions for the array.
            max_dimension = np.min(np.array([len(shape) for shape in shapes]))
            while dimension < max_dimension:
                if not np.all([
                    shape[:dimension + 1] == shapes[0][:dimension + 1]
                    for shape in shapes[1:]]):
                    break
                dimension += 1
        else:
            dimension = 0
        rows_shape = shapes[0][:dimension]
    else:
        # Check for consistent shapes of the columns and determine the
        # shape of the rows for the new tabular object.
        rows_shape = shapes[0][:dimension]
        if not np.all([shape[:dimension] == rows_shape
                       for shape in shapes[dimension:]]):
            raise ValueError('Column arrays have inconsistent shapes.')

    if ttype == 'table':
        # Astropy table.
        if dimension != 1:
            raise ValueError('Row shape must be 1D for astropy table.')
        result = tabular.copy(copy_data=False)
        result._init_from_cols(columns.values())
        return result

    if ttype == 'numpy':
        # Numpy structured array or astropy Quantity.
        dtype = []
        for name, shape in zip(columns, shapes):
            if len(shape) > dimension:
                dtype.append((name, columns[name].dtype, shape[dimension:]))
            else:
                dtype.append((name, columns[name].dtype))
        # Create an empty array with the necessary columns.
        if numpy.ma.isMaskedArray(tabular):
            result = numpy.ma.empty(rows_shape, dtype)
        else:
            result = np.empty(rows_shape, dtype)
        # Copy the column data into the newly created structured array.
        for name in columns:
            result[name] = columns[name]
        # Copy any units.
        try:
            result = result * tabular.unit
        except AttributeError:
            pass
        return result

    assert ttype == 'dict'
    # Dictionary of array-like objects.
    result = tabular.__class__()
    for name in columns:
        result[name] = columns[name]
    return result


def prepare_data(mode, args, kwargs):
    """Prepare data for an algorithm.

    Parameters
    ----------
    mode : 'in_place' or 'read_only'
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
        an astropy table, numpy structured array or arrays dictionary.
        The original column order will be preserved for an input tabular
        object by returning a collection.OrderedDict.
    """
    if mode not in ('in_place', 'read_only'):
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
        ttype = get_tabular_type(tabular)
        # Preserve the order of columns in the input table.
        arrays = collections.OrderedDict()
        if ttype == 'table':
            for name in tabular.colnames:
                arrays[name] = tabular[name]
        elif ttype == 'numpy':
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
            if mode == 'in_place':
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
