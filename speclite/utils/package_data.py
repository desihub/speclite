import os
from importlib import resources

# TODO: should make these Path objects

def get_path_of_data_file(data_file):
    """convenience wrapper to return location of data file
    """
    return os.path.join(str(get_path_of_data_dir()), data_file)


def get_path_of_data_dir():
    """convenience wrapper to return location of data directory
    """
    return resources.files('speclite') / 'data'
