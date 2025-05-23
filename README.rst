================
speclite package
================

|Astropy| |PyPI| |License| |Actions Status| |Coveralls Status| |Documentation Status|

Introduction
------------

This package provides a set of lightweight utilities for working with spectroscopic data in astronomy.

* Free software: 3-clause BSD style license.
* Documentation: `latest <http://speclite.readthedocs.io/en/latest/>`_ | `stable <http://speclite.readthedocs.io/en/stable/>`_
* Based on the Astropy affiliated package template.

Citation
--------

|DOI|

Please cite as:

David Kirkby, *et al.*. (2024).
desihub/speclite: Bug fix release: General clean-up prior to refactoring package infrastructure (v0.20). Zenodo. https://doi.org/10.5281/zenodo.13225530

See also the file `CITATION.cff`.

Requirements
------------

Speclite has the following package requirements:

* `NumPy <https://numpy.org/>`__
* `SciPy <https://scipy.org/>`__
* `Matplotlib <https://matplotlib.org>`__
* `Astropy <https://www.astropy.org/>`__
* `PyYAML <https://pyyaml.org>`__

License
-------

speclite is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.

Releasing
---------

Please follow these instructions when creating a new tag of speclite.

1. Update ``docs/changes.rst``: set the date corresponding to the next tag.
2. Check in the changes; a ``git push`` is optional at this point.
3. Create the tag: ``git tag -s -m 'Tagging speclite/vX.Y.Z' vX.Y.Z``. ``-s`` is optional; it adds a cryptographic signature to the tag.
4. Update ``docs/changes.rst``: add a new entry for a future tag with ``(unreleased)``.
5. Check in the changes, then push: ``git push; git push --tags``.
6. In your git clone, check out the tag: ``git co vX.Y.Z``.
7. Run ``python -m build``. This will create tar and wheel files in the ``dist/`` directory.
8. In the ``dist/`` directory, inspect the ``.tar.gz`` file. Make sure the version is set properly, that all expected files are present, etc.
9. In the ``dist/`` directory, run ``python -m twine upload *.gz *.whl``.
10. In your git clone, clean up and go back to ``main``.  You don't want to accidentally edit or commit on a tag.
11. On GitHub, create a new Release corresponding to the tag.  This is important: creating a release will also automatically create a new DOI on Zenodo.
12. On the ``main`` branch, update the ``README.rst`` file (this file), and the ``CITATION.cff`` file with the new DOI.

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13225530.svg
    :target: https://doi.org/10.5281/zenodo.13225530
    :alt: DOI: 10.5281/zenodo.13225530

.. |Astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. |License| image:: https://img.shields.io/pypi/l/speclite.svg
    :target: https://pypi.org/project/speclite/
    :alt: BSD License

.. |Actions Status| image:: https://github.com/desihub/speclite/workflows/CI/badge.svg
    :target: https://github.com/desihub/speclite/actions
    :alt: GitHub Actions CI Status

.. |Coveralls Status| image:: https://coveralls.io/repos/desihub/speclite/badge.svg?branch=main
    :target: https://coveralls.io/github/desihub/speclite?branch=main
    :alt: Test Coverage Status

.. |Documentation Status| image:: https://readthedocs.org/projects/speclite/badge/?version=latest
    :target: https://speclite.readthedocs.org/en/latest/
    :alt: Documentation Status

.. |PyPI| image:: https://img.shields.io/pypi/v/speclite.svg
    :target: https://pypi.org/project/speclite/
    :alt: Distribution Status
