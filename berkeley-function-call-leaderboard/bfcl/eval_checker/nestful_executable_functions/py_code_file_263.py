from distutils.version import LooseVersion

from typing import Dict



def find_highest_version_package(package_dict: Dict[str, str]) -> str:

    """Finds the name of the package with the highest version number in a dictionary of package names and version strings.

    Args:

        package_dict: A dictionary of package names and version strings.

    Returns:

        The name of the package with the highest version number, or None if the dictionary is empty.

    """

    if not package_dict:

        return None



    highest_version_package = None

    highest_version = LooseVersion('0.0.0')



    for package_name, version_str in package_dict.items():

        if version_str:

            version = LooseVersion(version_str)

            if version > highest_version:

                highest_version = version

                highest_version_package = package_name



    return highest_version_package

