# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Apply redshift transformations to wavelength, flux, inverse variance, etc.
"""
from __future__ import print_function, division

import numpy as np


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


def prepare_data(mode, args, kwargs):
    """Prepare data for an algorithm.

    Parameters
    ----------
    mode : tuple or str
        Either the desired shape of each array or else one of the strings
        'in_place' or `read_only`.  When a shape tuple is specified, the
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
    dict
        A dictionary of array names and corresponding numpy unstructured
        arrays.
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
                        units=c.units, meta=c.meta,
                        data=np.empty(output_shape, dtype=c.dtype))
                    tabular.replace_column(name, new_column)
            data = {name: tabular[name] for name in tabular.colnames}
        # Test for a numpy structured array.
        elif (hasattr(tabular, 'dtype') and
              getattr(tabular.dtype, 'names', None) is not None):
            if output_shape is not None:
                # Make a copy of this structured array.
                tabular = tabular.copy()
                # Change the shape of the new copy.
                tabular.resize(output_shape)
            data = {name: tabular[name] for name in tabular.dtype.names}
        elif tabular is not None:
            raise ValueError(
                'Cannot prepare input from tabular type {0}.'
                .format(type(tabular)))

    else:
        # Each value must be convertible to an unstructured numpy array.
        data = {}
        for name in kwargs:
            if output_shape is not None:
                data[name] = np.asanyarray(kwargs[name]).copy()
                data[name].resize(output_shape)
            elif mode == 'in_place':
                # This will fail unless the array is already an instance
                # of a numpy array.
                try:
                    data[name] = kwargs[name].view()
                except AttributeError:
                    raise ValueError('Cannot update array "{0}" in place.'
                                     .format(name))
            else:
                # Create a view with no memory copy when possible.
                data[name] = np.asanyarray(kwargs[name])
            # Check that this is not a structured array.
            if data[name].dtype.names is not None:
                raise ValueError(
                    'Cannot pass structured array "{0}" as keyword arg.'
                    .format(name))

    # Verify that all values are subclasses of np.ndarray with the same shape
    # and create read-only views if requested.
    for i, name in enumerate(data):
        if not isinstance(data[name], np.ndarray):
            raise RuntimeError(
                'Data for "{0}" has invalid type {1}.'.format(name, type(data[name])))
        if i == 0:
            input_shape = data[name].shape
        elif data[name].shape != input_shape:
            raise ValueError('Input arrays have different shapes.')
        if data[name].flags.writeable and mode == 'read_only':
            data[name] = data[name].view()
            data[name].flags.writeable = False

    return data
