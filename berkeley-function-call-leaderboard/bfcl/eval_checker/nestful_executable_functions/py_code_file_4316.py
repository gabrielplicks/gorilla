from typing import Tuple



def increase_version_number(version_number: str) -> str:

    """Increments the version number of a software program by 1.



    The version number is given as a string of the format `major.minor.patch`,

    where each part is a non-negative integer.



    Args:

        version_number: The version number of the software program.



    Returns:

        The incremented version number as a string.

    """

    major, minor, patch = version_number.split('.')

    major = int(major)

    minor = int(minor)

    patch = int(patch)

    patch += 1

    if patch == 256:

        patch = 0

        minor += 1

        if minor == 256:

            minor = 0

            major += 1

    new_version_number = f"{major}.{minor}.{patch}"



    return new_version_number

