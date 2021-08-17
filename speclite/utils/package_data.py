import os
from pathlib import Path

import pkg_resources


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename(
        "speclite", f"data/{data_file}")

    return Path(file_path)


def get_path_of_data_dir() -> Path:
    file_path = pkg_resources.resource_filename("speclite", "data")

    return Path(file_path)
