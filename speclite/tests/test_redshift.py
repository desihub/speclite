# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..redshift import transform, apply_redshift_transform
import numpy as np
import numpy.ma as ma
import astropy.table
import astropy.units as u


def test_z_par():
    # Must either provide z or both z_in and z_out.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'flux'))
    with pytest.raises(ValueError):
        transform(t)
    with pytest.raises(ValueError):
        transform(t, z=1, z_out=0)
    with pytest.raises(ValueError):
        transform(t, z=1, z_in=0)
    with pytest.raises(ValueError):
        transform(t, z_in=0)
    with pytest.raises(ValueError):
        transform(t, z_out=0)
    # Values must be > -1.
    with pytest.raises(ValueError):
        transform(t, z=-1)
    with pytest.raises(ValueError):
        transform(t, z_in=-1, z_out=0)
    with pytest.raises(ValueError):
        transform(t, z_in=0, z_out=-1)
    # Passing z is equivalent to passing z_out=z and z_in=0.
    t2 = transform(t, z=1.)
    t3 = transform(t, z_in=0., z_out=1.)
    assert np.all(t2['wlen'] == t3['wlen'])
    assert np.all(t2['flux'] == t3['flux'])


def test_table():
    # Exponents are inferred from column names.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'flux'))
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Transforms can be applied in place.
    tc = transform(t, z=1., in_place=True)
    assert tc is t
    assert np.all(t['wlen'] == t2['wlen'])
    assert np.all(t['flux'] == t2['flux'])
    # Unrecognized names are passed through.
    t = astropy.table.Table([[1.,2.,3.], [1.,1.,1.]], names=('wlen', 'extra'))
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['extra'] == t['extra'])
    # Any input units are preserved.
    t = astropy.table.Table()
    t['wlen'] = np.array([1., 2., 3.]) * u.Angstrom
    t['flux'] = np.ones(3) * u.erg / (u.s * u.cm ** 2 * u.Angstrom)
    t2 = transform(t, z=1.)
    assert t2['wlen'].unit == t['wlen'].unit
    assert t2['flux'].unit == t['flux'].unit
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Any input masks are preserved.
    t = astropy.table.Table([[1., 2., 3.], [1., 1., 1.]],
                            names=('wlen', 'flux'), masked=True)
    t['flux'].mask[1] = True
    t2 = transform(t, z=1.)
    assert np.all(t['flux'].mask == t2['flux'].mask)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Multiple redshifts can be applied using broadcasting although this
    # is somewhat awkward for tables because the first shape dimension
    # corresponds to the number of rows, which cannot change.
    # In this example, each table row contains fluxes for one spectrum
    # and we apply a different redshift to each spectrum.
    t = astropy.table.Table([[[1., 1., 1.], [1., 1., 1.]]], names=('flux',))
    assert t['flux'].shape == (2, 3)
    tm = transform(t, z=np.array([[1.], [3.]]))
    assert tm['flux'].shape == (2, 3)
    assert np.all(tm['flux'][0] == t['flux'][0] / 2.)
    assert np.all(tm['flux'][1] == t['flux'][1] / 4.)
    # Since the shape does not change, this example can also be done in place.
    transform(t, z=np.array([[1.], [3.]]), in_place=True)


def test_structured_array():
    # Exponents are inferred from column names.
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Transforms can be applied in place.
    tc = transform(t, z=1., in_place=True)
    assert tc is t
    assert np.all(t['wlen'] == t2['wlen'])
    assert np.all(t['flux'] == t2['flux'])
    # Unrecognized names are passed through.
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('extra', float)])
    t2 = transform(t, z=1.)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['extra'] == t['extra'])
    # Any input units are preserved.
    t = np.array([(1,), (2,), (3,)], dtype=[('wlen', float)]) * u.Angstrom
    t2 = transform(t, z=1.)
    assert t2['wlen'].unit == t['wlen'].unit
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    # Any input masks are preserved.
    t = ma.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    t['flux'].mask[1] = True
    t2 = transform(t, z=1.)
    assert np.all(t['flux'].mask == t2['flux'].mask)
    assert np.all(t2['wlen'] == 2 * t['wlen'])
    assert np.all(t2['flux'] == 0.5 * t['flux'])
    # Multiple redshifts can be applied using broadcasting. Note that
    # you are responsible for providing redshifts with a suitable shape
    # for broadcasting.
    t = np.array([(1, 1), (2, 1), (3, 1)],
                 dtype=[('wlen', float), ('flux', float)])
    assert t.shape == (3,)
    tm = transform(t, z=np.array([[1.], [3.]]))
    assert tm.shape == (2, 3)
    assert np.all(tm['wlen'][0] == 2 * t['wlen'])
    assert np.all(tm['flux'][0] == 0.5 * t['flux'])
    # When broadcasting changes the shape, the transform cannot be
    # performed in place.
    with pytest.raises(ValueError):
        transform(t, z=np.array([[1.], [3.]]), in_place=True)


def test_arrays_dict():
    # Exponents are inferred from array names.
    d = transform(wlen=[1., 2., 3.], flux=[1., 1., 1.], z=1.)
    assert np.all(d['wlen'] == [2., 4., 6.])
    assert np.all(d['flux'] == [0.5, 0.5, 0.5])
    # Transforms can be applied in place when inputs are numpy arrays.
    d = dict(wlen=np.array([1., 2., 3.]), flux=np.array([1., 1., 1.]))
    transform(z=1., in_place=True, **d)
    assert np.all(d['wlen'] == [2., 4., 6.])
    assert np.all(d['flux'] == [0.5, 0.5, 0.5])
    # Unrecognized names are passed through.
    d = transform(wlen=[1., 2., 3.], extra=[1., 1., 1.], z=1.)
    assert np.all(d['wlen'] == [2., 4., 6.])
    assert np.all(d['extra'] == [1., 1., 1.])
    # Any input units are preserved.
    flux_unit = u.erg / (u.s * u.cm ** 2 * u.Angstrom)
    d = transform(z=1, wlen=[1., 2., 3.] * u.Angstrom,
                  flux=[1., 1., 1.] * flux_unit)
    assert np.all(d['wlen'] == [2., 4., 6.] * u.Angstrom)
    assert np.all(d['flux'] == [0.5, 0.5, 0.5] * flux_unit)
    # Any input masks are preserved.
    d = dict(wlen=ma.array([1., 2., 3.]), flux=[1., 1., 1.])
    d['wlen'].mask = [False, True, False]
    d2 = transform(z=1., **d)
    assert np.all(d['wlen'].mask == d2['wlen'].mask)
    assert np.all(d2['wlen'][~d2['wlen'].mask].filled() == [2., 6.])
    assert np.all(d2['flux'] == [0.5, 0.5, 0.5])
    # Multiple redshifts can be applied using broadcasting. Note that
    # you are responsible for providing redshifts with a suitable shape
    # for broadcasting.
    d = transform(wlen=[1., 2., 3.], flux=[1., 1., 1.], z=[[1.], [3.]])
    assert np.all(d['wlen'] == [[2., 4., 6.], [4., 8., 12.]])
    assert np.all(d['flux'] == [[.5, .5, .5], [.25, .25, .25]])
