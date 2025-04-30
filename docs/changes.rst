===================
speclite Change Log
===================

1.0.0 (unreleased)
------------------

- Eliminate ``astropy_helpers``; fully modern package layout (PR `#97`_).
- Support Numpy 2.0 (PR `#97`_).
- Add Intermediate-Band Imaging Survey (IBIS) filters (PR `#94`_).

.. _`#94`: https://github.com/desihub/speclite/pull/94
.. _`#97`: https://github.com/desihub/speclite/pull/97

0.20 (2024-08-05)
-----------------

- Fix ``scipy.integrate`` name changes; pin Pillow version so unit tests work (PR `#91`_).
- Fix issues related to :meth:`~specutils.filters.FilterSequence.pad_spectrum` and
  :meth:`~specutils.filters.FilterResponse.get_ab_maggies`; allow ``pip install``
  on more recent Python versions;
  replace deprecated ``matplotlib`` colormap function (PR `#92`_).

.. _`#91`: https://github.com/desihub/speclite/pull/91
.. _`#92`: https://github.com/desihub/speclite/pull/92

0.19 (2024-04-30)
-----------------

- Add LSST 2023 filters (silver-coated mirrors) (PR `#84`_).

.. _`#84`: https://github.com/desihub/speclite/pull/84

0.18 (2024-01-16)
-----------------

- Add additional filters: GALEX, CFHT-MegaCam, Suprime-Cam intermediate-band,
  and ODIN narrow-band (PR `#82`_).
- Add Gaia DR3/EDR3 filters (PR `#83`_).

.. _`#82`: https://github.com/desihub/speclite/pull/82
.. _`#83`: https://github.com/desihub/speclite/pull/83

0.17 (2023-09-14)
-----------------

- Add 2MASS JHKs filter curves (PR `#79`_).
- Restore some missing CI tests and other infrastructure updates (PR `#81`_).

.. _`#79`: https://github.com/desihub/speclite/pull/79
.. _`#81`: https://github.com/desihub/speclite/pull/81

0.16 (2022-07-19)
-----------------

- Add new SDSS filters which include atmospheric extinction (airmass=1.3) (PR `#76`_).

.. _`#76`: https://github.com/desihub/speclite/pull/76

0.15 (2022-01-10)
-----------------

- Fix documentation builds and other deprecation warnings (PR `#73`_, `#72`_).

.. _`#73`: https://github.com/desihub/speclite/pull/73
.. _`#72`: https://github.com/desihub/speclite/pull/72

0.14 (2021-09-09)
-----------------

- Update SDSS filter metadata to reflect that no atmospheric extinction is included (602e805_).
- Fix deprecated Astropy utilities (PR `#68`_).
- Migrate to GitHub Actions (PR `#65`_).

.. _602e805: https://github.com/desihub/speclite/commit/602e80562615c11e86429576b2f9b996efe39050
.. _`#68`: https://github.com/desihub/speclite/pull/68
.. _`#65`: https://github.com/desihub/speclite/pull/65

0.13 (2021-01-18)
-----------------

- Add GAIA DR2 filter curves (PR `#62`_).

.. _`#62`: https://github.com/desihub/speclite/pull/62

0.12 (2021-01-15)
-----------------

- Add DECam DR1 filter curves with and without X=1.2 atmosphere (PR `#61`_).

.. _`#61`: https://github.com/desihub/speclite/pull/61

0.11 (2020-11-23)
-----------------

- Another attempt to fix broken ``astropy_helpers`` in PyPI release.

0.10 (2020-11-23)
-----------------

- Fix broken 0.9 PyPI release.

0.9 (2020-07-31)
----------------

- Minor updates for Python 3.8 support (PR `#55`_).
- Fix installation problems (`#56`_).

.. _`#56`: https://github.com/desihub/speclite/pull/56
.. _`#55`: https://github.com/desihub/speclite/pull/55

0.8 (2018-09-11)
----------------

- Add MzLS-z, BASS-g, BASS-r filters used by DESI imaging surveys.
- Update ``astropy_helpers`` to v2.0.6.

0.7 (2017-10-03)
----------------

- Update to ``astropy_helpers`` v2.0.1 (PR `#32`_).

.. _`#32`: https://github.com/desihub/speclite/pull/32

0.6 (2017-10-02)
----------------

- Remove tests against Python 3.3 and Numpy 1.7.
- Add HSC filters (PR `#31`_).
- Save filter curves at full machine precision (PR `#27`_).
- Update to latest Astropy affiliate package template.
- Add support and travis testing for Python 3.6.
- Add support for band shifted filters (PR `#20`_).
- Clean up package-level imports (PR `#23`_).

.. _`#31`: https://github.com/desihub/speclite/pull/31
.. _`#27`: https://github.com/desihub/speclite/pull/27
.. _`#23`: https://github.com/desihub/speclite/pull/23
.. _`#20`: https://github.com/desihub/speclite/pull/20

0.5 (2016-08-22)
----------------

- Update to latest Astropy affiliate package template.
- Drop support for Python 2.6 and add support for Python 3.5.
- Add testing against LTS release of Astropy.
- Drop testing against Numpy 1.6 and add Numpy 1.11.
- Update ReadTheDocs URLs (.org -> .io).
- Add LSST filter response curves.

0.4 (2016-02-17)
----------------

- Improve filter module error messages and validation.
- Add mask_invalid option to FilterSequence methods.
- Implement pad_spectrum method of FilterResponse and FilterSequence.

0.3 (2016-02-05)
----------------

- Add filter response curves and filters module.
- Add pyyaml required dependency (for reading filter curves).

0.2 (2015-11-15)
----------------

- Add downsample module.

0.1 (2015-08-05)
----------------

- Initial release.
