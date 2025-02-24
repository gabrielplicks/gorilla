import re

import typing



def is_valid_version_number(version: str) -> bool:

    """Checks if a given string is a valid version number.



    A version number is valid if it follows the format "major.minor.patch", where

    major, minor, and patch are all non-negative integers separated by dots, and

    patch is optional.



    Args:

        version: The string to check.



    Returns:

        A boolean value indicating whether the version number is valid.

    """

    pattern = r"^\d+(?:\.\d+){,2}$"

    return bool(re.match(pattern, version))

