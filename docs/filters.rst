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
