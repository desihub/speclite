import os

_has_importlib = True
try:
    from importlib.resources import files
    resource_filename = None
except ImportError:
    from pkg_resources import resource_filename
    _has_importlib = False

# TODO: should make these Path objects

def get_path_of_data_file(data_file):
    """convenience wrapper to return location of data file
    """
    return os.path.join(get_path_of_data_dir(), data_file)


def get_path_of_data_dir():
    """convenience wrapper to return location of data directory
    """
    if _has_importlib:
        return str(files('speclite') / 'data')
    return resource_filename('speclite', 'data')
