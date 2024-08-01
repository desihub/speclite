import os
import pathlib
from ..utils.package_data import get_path_of_data_file, get_path_of_data_dir


def test_get_path_of_data_file():
    data_file = get_path_of_data_file('filters/twomass-Ks.ecsv')
    assert os.path.exists(data_file)


def test_get_path_of_data_dir():
    data_dir = get_path_of_data_dir()
    assert isinstance(data_dir, pathlib.PosixPath)
    assert os.path.isdir(data_dir)
