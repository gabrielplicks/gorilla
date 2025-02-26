from typing import Union



def format_filesize(filesize: Union[int, float]) -> str:

    """Formats a filesize into a human-readable string representation.



    Args:

        filesize: The filesize in bytes.



    Returns:

        A string in the format "{filesize} {unit}", where the unit is the appropriate unit for the filesize.

        If the filesize is zero or negative, the output is "0 B".

    """

    if filesize <= 0:

        return "0 B"



    units = ["B", "kB", "MB", "GB"]

    for unit in units:

        if filesize < 1024:

            break

        filesize /= 1024



    return f"{filesize:.1f} {unit}"

