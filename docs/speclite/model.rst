Data Model
==========

The design goals of this package are to:
 * Use spectral data "as-is" rather than requiring users to repackage their spectra into objects defined by this package.
 * Minimize the number of assumptions about how a user's spectral data is organized.
 * Limit the required dependencies to `astropy <http://astropy.readthedocs.org/>`__ and its dependencies.
 * Use the `astropy affiliated package template <https://github.com/astropy/package-template>`__.

We assume that, at a minimum, a spectrum is defined by an array of wavelengths and an array of fluxes.
