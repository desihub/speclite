import os

import pkg_resources

# TODO: should make these Path objects

def get_path_of_data_file(data_file) -> str:
    """convenience wrapper to return location of data file
    """
    file_path = pkg_resources.resource_filename(
        "speclite", os.path.join("data", f"{data_file}"))

    return file_path


def get_path_of_data_dir() -> str:
    """convenience wrapper to return location of data directory
    """
    file_path = pkg_resources.resource_filename("speclite", "data")

    return file_path
