import os
from ..utils import package_data


def test_get_path_of_data_file():
    data_file = package_data.get_path_of_data_file('filters/twomass-Ks.ecsv')
    assert os.path.exists(data_file)


def test_get_path_of_data_dir():
    data_dir = package_data.get_path_of_data_dir()
    assert os.path.isdir(data_dir)


def test_get_path_of_data_dir_no_importlib(monkeypatch):
    data_dir = package_data.get_path_of_data_dir()
    def mock_resource(foo, bar):
        return data_dir
    monkeypatch.setattr(package_data, '_has_importlib', False)
    monkeypatch.setattr(package_data, 'resource_filename', mock_resource)
    data_dir2 = package_data.get_path_of_data_dir()
    assert data_dir2 == data_dir

