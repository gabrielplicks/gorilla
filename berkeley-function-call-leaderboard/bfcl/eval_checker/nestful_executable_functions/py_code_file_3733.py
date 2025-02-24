from typing import Union



def human_readable_size(size_bytes: Union[int, float]) -> str:

    """Converts a given number of bytes to a human-readable format.



    Args:

        size_bytes: The size in bytes to be converted.



    Returns:

        The formatted size as a string.

    """

    units = ['B', 'KB', 'MB', 'GB', 'TB']

    if size_bytes == 0:

        return "0B"  # Directly return for 0 bytes to avoid division by zero in loop

    num = abs(size_bytes)

    unit = units[0]



    for u in units[1:]:

        if num < 1024.0:

            break

        num /= 1024.0

        unit = u



    formatted_size = f"{num:.1f}{unit}"

    return formatted_size if size_bytes >= 0 else f"-{formatted_size}"

