import numpy as np
import numpy.ma as ma


def transform(z_in, z_out, data_in=None, data_out=None, rules={}):
    """
    Transform spectral data from redshift z_in to z_out.

    Each quantity X is transformed according to a power law::

        X_out = X_in * ((1 + z_out) / (1 + z_in))**exponent

    where exponents are specified with the ``rules`` argument. Exponents for
    some common cases are listed in the table below.

    ======== ================================================================
    Exponent Quantities
    ======== ================================================================
    0        flux density in photons/(s*cm^2*Ang)
    +1       wavelength, wavelength error, flux density in ergs/(s*cm^2*Hz)
    -1       frequency, frequency error, flux density in ergs/(s*cm^2*Ang)
    +2       inverse variance of flux density in ergs/(s*cm^2*Ang)
    -2       inverse variance of flux density in ergs/(s*cm^2*Hz)
    ======== ================================================================

    For example, to transform separate wavelength and flux arrays using the
    SDSS standard units of Ang and 1e-17 erg/(s*cm^2*Ang):

    >>> wlen = np.arange(4000., 10000.)
    >>> flux = np.ones(wlen.shape)
    >>> result = transform(z_in=0, z_out=1, rules=[
    ... dict(name='wlen', exponent=+1, array_in=wlen),
    ... dict(name='flux', exponent=-1, array_in=flux)])
    >>> result.dtype
    dtype([('wlen', '<f8'), ('flux', '<f8')])
    >>> result['flux'][:5]
    array([ 0.5,  0.5,  0.5,  0.5,  0.5])

    The same calculation could be performed with the input data stored in
    a numpy structured array, in which case any additional fields are
    copied to the output array:

    >>> data = np.empty(6000, dtype=[
    ... ('wlen', float), ('flux', float), ('maskbits', int)])
    >>> data['wlen'] = np.arange(4000., 10000.)
    >>> data['flux'] = np.ones_like(data['wlen'])
    >>> result = transform(z_in=0, z_out=1, data_in=data, rules=[
    ... dict(name='wlen', exponent=+1),
    ... dict(name='flux', exponent=-1)])
    >>> result.dtype
    dtype([('wlen', '<f8'), ('flux', '<f8'), ('maskbits', '<i8')])
    >>> result['flux'][:5]
    array([ 0.5,  0.5,  0.5,  0.5,  0.5])

    The transformed result is always a `numpy structured array
    <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__, with field
    (column) names determined by the rules you provide. Use dictionary
    syntax to access individual fields, as in the examples above. You can
    also use the more convenient dot notation using `this recarray recipe
    <http://wiki.scipy.org/Cookbook/Recarray>`__, although this is generally
    slower so results are not automatically converted to `numpy recarrays
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`__
    .

    The usual `numpy broadcasting rules
    <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__ apply
    in the transformation expression above so, for example, the same redshift
    can be applied to multiple spectra, or different redshifts can be applied
    to the same spectrum with appropriate input shapes.

    Input arrays can have associated `masks
    <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ and these
    will be propagated to the output. Input arrays can also have `units
    <http://astropy.readthedocs.org/en/latest/units/index.html>`__ but these
    will not be used or propagated to the output since numpy structured arrays
    do not support per-column units.

    Parameters
    ----------
    z_in: float or numpy.ndarray
        Redshift(s) of the input spectral data, which must all be >= 0.
    z_out: float or numpy.ndarray
        Redshift(s) of the output spectral data, which must all be >= 0.
    data_in: numpy.ndarray
        Structured numpy array containing input spectrum data to transform. If
        none is specified, then all quantities must be provided as numpy arrays
        in the rules.
    data_out: numpy.ndarray
        Structured numpy array where output spectrum data should be written. If
        none is specified, then an appropriately sized array will be allocated
        and returned. Use this method to take control of the memory allocation
        and, for example, re-use the same output array for a sequence of
        transforms.
    rules: iterable
        An iterable object whose elements are dictionaries. Each dictionary
        specifies how one quantity will be transformed and must contain 'name'
        and 'exponent' values. If an 'array_in' value is also specified, it
        should refer to a numpy array containing the input values to transform.
        Otherwise, ``data_in[<name>]`` is assumed to contain the input values
        to transform.  If no ``rules`` are specified and ``data_in`` is
        provided, then ``data_out`` is just a copy of ``data_in``.

    Returns
    -------
    result: numpy.ndarray
        Array of spectrum data with the redshift transform applied. Equal to
        data_out when set, otherwise a new array is allocated. If ``data_in``
        is specified, then any fields not listed in ``rules`` are copied to
        ``data_out``, so effectively have an implicit exponent of zero.
    """

    if not isinstance(z_in, np.ndarray):
        z_in = np.float(z_in)
    if np.any(z_in < 0):
        raise ValueError('Found invalid z_in < 0.')
    if not isinstance(z_out, np.ndarray):
        z_out = np.float(z_out)
    if np.any(z_out < 0):
        raise ValueError('Found invalid z_out < 0.')
    z_factor = (1.0 + z_out) / (1.0 + z_in)

    if data_in is not None and not isinstance(data_in, np.ndarray):
        raise ValueError('Invalid data_in type: {0}.'.format(type(data_in)))
    if data_out is not None and not isinstance(data_out, np.ndarray):
        raise ValueError('Invalid data_out type: {0}.'.format(type(data_out)))

    if data_in is not None:
        shape_in = data_in.shape
        dtype_in = data_in.dtype
        masked_in = ma.isMA(data_in)
    else:
        shape_in = None
        dtype_in = []
        masked_in = False

    for i, rule in enumerate(rules):
        name = rule.get('name')
        if not isinstance(name, basestring):
            raise ValueError('Invalid name in rule: {0}'.format(name))
        try:
            exponent = np.float(rule.get('exponent'))
        except TypeError:
            raise ValueError(
                'Invalid exponent for {0}: {1}.'
                .format(name, rule.get('exponent')))
        if data_in is not None and name not in dtype_in.names:
            raise ValueError('No such data_in field named {0}.'.format(name))
        if data_out is not None and name not in data_out.dtype.names:
            raise ValueError('No such data_out field named {0}.'.format(name))
        array_in = rule.get('array_in')
        if array_in is not None:
            if data_in is not None:
                raise ValueError(
                    'Cannot specify data_in and array_in for {0}.'
                    .format(name))
            if not isinstance(array_in, np.ndarray):
                raise ValueError(
                    'Invalid array_in type for {0}: {1}.'
                    .format(name, type(array_in)))
            if shape_in is None:
                shape_in = array_in.shape
            elif shape_in != array_in.shape:
                raise ValueError(
                    'Incompatible array_in shape for {0}: {1}. Expected {2}.'
                    .format(name, array_in.shape, shape_in))
            dtype_in.append((name, array_in.dtype))
            if ma.isMA(array_in):
                masked_in = True
        else:
            if data_in is None:
                raise ValueError(
                    'Missing array_in for {0} (with no data_in).'.format(name))
            # Save a view of the input data column associated with this rule.
            rules[i]['array_in'] = data_in[name]

    shape_out = np.broadcast(np.empty(shape_in), z_factor).shape
    if data_out is None:
        if masked_in:
            data_out = ma.empty(shape_out, dtype=dtype_in)
            data_out.mask = False
        else:
            data_out = np.empty(shape_out, dtype=dtype_in)
    else:
        if masked_in and not ma.isMA(data_out):
            raise ValueError('data_out discards data_in mask.')
        if data_out.shape != shape_out:
            raise ValueError(
                'Invalid data_out shape: {0}. Expected {1}.'
                .format(data_out.shape, shape_out))
        if data_out.dtype != dtype_in:
            raise ValueError(
                'Invalid data_out dtype: {0}. Expected {1}.'
                .format(data_out.dtype, dtype_in))

    if data_in is not None:
        # Copy data_in to data_out so that any columns not listed in the
        # rules are propagated to the output.
        data_out[...] = data_in

    for rule in rules:
        name = rule.get('name')
        exponent = np.float(rule.get('exponent'))
        array_in = rule.get('array_in')
        data_out[name][:] = array_in * z_factor**exponent
        if data_in is None and ma.isMA(array_in):
            data_out[name].mask[...] = array_in.mask

    return data_out
