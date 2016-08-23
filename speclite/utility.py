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


def prepare_data(*args, **kwargs):
    """Prepare data for an algorithm.

    Parameters
    ----------
    *args
        A single non-keyword argument must be a tabular object that is
        internally organized into named arrays.  Instances of a numpy
        structured array, possibly including astropy units or a mask,
        and instances of an astropy table are both supported.
    **kwargs
        When only keyword arguments are present, each one is assumed to
        name a value that is convertible to a numpy (unstructured) array.
        The names ``always_copy`` and ``never_copy`` are reserved and
        interpreted as option values that control whether a copy is
        required or forbidden in the return value.

    Returns
    -------
    dict
        A dictionary of array names and corresponding numpy unstructured
        arrays. The arrays will use new memory when ``always_copy``
        is set and will re-use the input memory (or fail) when
        ``never_copy`` is set.  If neither is set, the input memory will
        be re-used if possible.
    """
    # Get the always_copy, never_copy option values.
    kwargs, options = get_options(kwargs, always_copy=False, never_copy=False)

    if len(args) > 1:
        raise ValueError('Must provide zero or one non-keyword args to prepare_data().')
    elif len(args) == 1 and len(kwargs) > 0:
        raise ValueError('Cannot provide both keyword and non-keyword args to prepare_data().')

    # A single non-keyword dictionary arg is converted into kwargs.
    if len(args) == 1 and isinstance(args[0], dict):
        kwargs = args[0]
        del args[0]

    if len(args) == 1:
        # The unique non-keyword argument must be a tabular type.
        tabular = args[0]
        # Test for an astropy.table.Table.  We check for the colnames attribute instead of
        # using isinstance() so that this test fails gracefully if astropy is not installed.
        if hasattr(tabular, 'colnames'):
            if options['always_copy']:
                # Make a copy of this table and its underlying data.
                tabular = tabular.copy(copy_data=True)
            data = {name: tabular[name] for name in tabular.colnames}
        # Test for a numpy structured array.
        elif hasattr(tabular, 'dtype') and getattr(tabular.dtype, 'names', None) is not None:
            if options['always_copy']:
                # Make a copy of this structured array.
                tabular = tabular.copy()
            data = {name: tabular[name] for name in tabular.dtype.names}
        elif tabular is not None:
            raise ValueError('Cannot prepare input from tabular type {0}.'.format(type(tabular)))

    else:
        # Each value must be convertible to a non-structured numpy array.
        data = {}
        for name in kwargs:
            if options['always_copy']:
                data[name] = np.array(kwargs[name])
            elif options['never_copy']:
                data[name] = kwargs[name].view()
            else:
                # Creates a view with no memory copy when possible.
                data[name] = np.asanyarray(kwargs[name])
            # Check that this is not a structured array.
            if data[name].dtype.names is not None:
                raise ValueError(
                    'Cannot pass structured array "{0}" as keyword arg to prepare_data().'
                    .format(name))

    # Verify that each value is a a subclass of np.ndarray.
    for name in data:
        if not isinstance(data[name], np.ndarray):
            raise RuntimeError(
                'Data for "{0}" has invalid type {1}.'.format(name, type(data[name])))

    return data
