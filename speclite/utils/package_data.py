import os

import pkg_resources

# TODO: should make these Path objects

def get_path_of_data_file(data_file) -> str:
    file_path = pkg_resources.resource_filename(
        "speclite", os.path.join("data", f"{data_file}"))

    return file_path


def get_path_of_data_dir() -> str:
    file_path = pkg_resources.resource_filename("speclite", "data")

    return file_path
