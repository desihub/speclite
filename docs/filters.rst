Filter Response Curves
======================

The ``data/filters/`` subdirectory contains small files with tabulated
filter response curves.  All files contain a single curve stored in the ASCII
`enhanced character-separated value format
<https://github.com/astropy/astropy-APEs/blob/master/APE6.rst>`__, which is
used to specify the wavelength units and provide the following metadata:

+------------+-------------------------------------------------------------------------+
|Key         | Description                                                             |
+============+=========================================================================+
|group_name  | Name of the group that this filter response belongs to, e.g., sdss2010. |
+------------+-------------------------------------------------------------------------+
|band_name   | Name of the filter pass band, e.g., r.                                  |
+------------+-------------------------------------------------------------------------+
|airmass     | Airmass used for atmospheric transmission, or zero for no atmosphere.   |
+------------+-------------------------------------------------------------------------+
|url         | URL with more details on how this filter response was obtained.         |
+------------+-------------------------------------------------------------------------+
|description | Brief description of this filter response.                              |
+------------+-------------------------------------------------------------------------+

The following sections summarize the standard filters included with the ``speclite``
code distribution.  Refer to the :ref:`filters module <filters-api>` API docs for
details and examples of how to load an use these filters.  See
:ref:`below <custom-filters>` for instructions on working with your own custom
filters.

SDSS Filters
------------

SDSS filter responses are taken from Table 4 of `Doi et al, "Photometric
Response Functions of the SDSS Imager", The Astronomical Journal, Volume 139,
Issue 4, pp. 1628-1648 (2010)
<http://dx.doi.org/10.1088/0004-6256/139/4/1628>`__, and calculated as the
reference response multiplied by the reference APO atmospheric transmission
at an airmass 1.3.  See the paper for details.

The group name ``sdss2010`` is used to identify these response curves in
``speclite``. The plot below shows the output of::

    sdss = speclite.filters.load_filters('sdss2010-*')
    speclite.filters.plot_filters(sdss, wavelength_limits=(3000, 11000))

.. image:: _static/sdss2010.png
    :alt: sdss2010 filter curves

DECam Filters
-------------

DECam filter responses are taken from this `Excel spreadsheet
<http://www.ctio.noao.edu/noao/sites/default/files/DECam/DECam_filters.xlsx>`__
created by William Wester in September 2014 and linked to this `NOAO DECam page
<http://www.ctio.noao.edu/noao/content/dark-energy-camera-decam>`__.
Throughputs include a reference atmosphere with airmass 1.3 provided by Ting Li.
These are the most recent publicly available DECam throughputs as of Feb 2016.

The group name ``decam2014`` is used to identify these response curves in
``speclite``. The plot below shows the output of::

    decam = speclite.filters.load_filters('decam2014-*')
    speclite.filters.plot_filters(
        decam, wavelength_limits=(3000, 11000), legend_loc='upper left')

.. image:: _static/decam2014.png
    :alt: decam2014 filter curves

WISE Filters
------------

WISE filter responses are taken from files linked to `this page
<http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA>`__
containing the weighted mean relative spectral responses described in
`Wright et al, "The Wide-field Infrared Survey Explorer (WISE): Mission Description
and Initial On-orbit Performance", The Astronomical Journal, Volume 140,
Issue 6, pp. 1868-1881 (2010)
<http://dx.doi.org/10.1088/0004-6256/140/6/1868>`__.

Note that these responses are based on pre-flight measurements but the in-flight
responses of the W3 and W4 filters are observed to have effective wavelengths
that differ by -(3-5)% and +(2-3)%, respectively.  Refer to Section 2.2 of
`Wright 2010 <http://dx.doi.org/10.1088/0004-6256/140/6/1868>`__ for details.
See also Section 2.1.3 of `Brown 2014
<http://dx.doi.org/10.1088/0067-0049/212/2/18>`__ for further details about W4.

The group name ``wise2010`` is used to identify these response curves in
``speclite``.  The plot below shows the output of the command below, and matches
Figure 6 of the paper::

    wise = speclite.filters.load_filters('wise2010-*')
    speclite.filters.plot_filters(wise, wavelength_limits=(2, 30),
        wavelength_unit=astropy.units.micron, wavelength_scale='log')
    plt.gca().set_xticks([2, 5, 10, 20, 30])
    plt.gca().set_xticklabels([2, 5, 10, 20, 30])

.. image:: _static/wise2010.png
    :alt: wise2010 filter curves

HSC Filters
-----------

HSC filter responses are taken from `here 
<https://hsc-release.mtk.nao.ac.jp/doc/index.php/survey/>`__. These
throughputs include a reference atmosphere with airmass 1.2. Refer to
Kawanamoto et al. 2017 (in prep).

The group name ``hsc2017`` is used to identify these curves in ``speclite``.
The plot below shows the output of the following command::

    hsc = speclite.filters.load_filters('hsc2017-*')
    speclite.filters.plot_filters(hsc)

.. image:: _static/hsc2017.png
    :alt: HSC filter curves

LSST Filters
------------

LSST filter responses are taken from tag 12.0 of the LSST simulations
`throughputs package <https://github.com/lsst/throughputs>`__ and include a
standard atmosphere with airmass 1.2 that is also tabulated in the same package.

The group name ``lsst2016`` is used to identify these response curves in ``speclite``.
The plot below shows the output of the command below, and matches
Figure 6 of the paper::

    lsst = speclite.filters.load_filters('lsst2016-*')
    speclite.filters.plot_filters(
        lsst, wavelength_limits=(3000, 11000), legend_loc='upper left')

.. image:: _static/lsst2016.png
    :alt: lsst filter curves

Johnson/Cousins Filters
-----------------------

Reference definitions of the Johnson/Cousins "standard" filters are taken
from Table 2 of `Bessell, M. S., "UBVRI passbands," PASP, vol. 102, Oct. 1990,
p. 1181-1199 <http://dx.doi.org/10.1086/132749>`__. We use the band name "U"
for the response that Table 2 refers to as "UX". Note that these do not
represent the response of any actual instrument. Response values are normalized
to have a maximum of one in each band.

The group name `bessell` is used to identify these response curves in
``speclite``.  The plot below shows the output of the command below::

    bessell = speclite.filters.load_filters('bessell-*')
    speclite.filters.plot_filters(bessell, wavelength_limits=(2900, 9300))

.. image:: _static/bessell.png
    :alt: bessell filter curves

.. _custom-filters:

Custom Filters
--------------

In addition to the standard filters included with the ``speclite`` code
distribution, you can create and read your own custom filters.  For example
to define a new filter group called "fangs" with "g" and "r" bands, you
will first need to define your filter responses with new
:class:`speclite.filters.FilterResponse` objects::

    fangs_g = speclite.filters.FilterResponse(
        wavelength = [3800, 4500, 5200] * u.Angstrom,
        response = [0, 0.5, 0], meta=dict(group_name='fangs', band_name='g'))
    fangs_r = speclite.filters.FilterResponse(
        wavelength = [4800, 5500, 6200] * u.Angstrom,
        response = [0, 0.5, 0], meta=dict(group_name='fangs', band_name='r'))

Your metadata dictionary must include the ``group_name`` and ``band_name``
keys, but all of the keys listed above are recommended. You can now load and
use these filters with their canonical names, although the group wildcard
``fangs-*`` is not supported.  For example::

    fangs = speclite.filters.load_filters('fangs-g', 'fangs-r')
    speclite.filters.plot_filters(fangs)

.. image:: _static/custom.png
    :alt: custom filter curves

Next, save these filters in the correct format to any directory::

    directory_name = '.'
    fg_name = fangs_g.save(directory_name)
    fr_name = fangs_r.save(directory_name)

Note that the file name in the specified directory is determined automatically
based on the filter's group and band names.

Finally, you can now read these custom filters from other programs by
calling :func:`speclite.filters.load_filter` with paths::

    directory_name = '.'
    fg_name = os.path.join(directory_name, 'fangs-g.ecsv')
    fr_name = os.path.join(directory_name, 'fangs-r.ecsv')
    fangs_g = speclite.filters.load_filter(fg_name)
    fangs_r = speclite.filters.load_filter(fr_name)

Note that :func:`load_filter <speclite.filters.load_filter>` and
:func:`load_filters <speclite.filters.load_filters>`
look for the ".ecsv" extension in the name to recognize a custom filter.
