from typing import Tuple



def load_software_version(version_file: str) -> str:

    """Loads the software version from a version file.



    The version file contains the version number and a list of commit hashes.

    The function returns a string in the format `vX.Y.Z-n`, where `X`, `Y`, and

    `Z` are the major, minor, and patch numbers, respectively, and `n` is the

    number of commits.



    Args:

        version_file: The path to the version file.



    Returns:

        The software version as a string.

    """

    with open(version_file, 'r') as f:

        version, *commits = f.read().splitlines()

    major, minor, patch = version.split('.')

    num_commits = len(commits)

    version_string = f'v{major}.{minor}.{patch}-{num_commits}'

    return version_string

