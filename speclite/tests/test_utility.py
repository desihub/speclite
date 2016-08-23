# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..utility import get_options
import numpy as np


def test_get_options_none():
    defaults = dict(x=1, y=2, z=3)
    kwin = dict(a=1, b=2, c=3)
    kwout, opt = get_options(kwin, **defaults)
    assert kwout == kwin
    assert opt == defaults


def test_get_options_some():
    defaults = dict(x=1, y=2, z=3)
    kwin = dict(a=1, b=2, c=3, x=-1)
    kwout, opt = get_options(kwin, **defaults)
    assert kwout == dict(a=1, b=2, c=3)
    assert opt == dict(x=-1, y=2, z=3)
