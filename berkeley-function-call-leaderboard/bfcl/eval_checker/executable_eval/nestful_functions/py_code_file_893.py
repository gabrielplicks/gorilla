from typing import Tuple



def parse_size(size_str: str) -> Tuple[int, int]:

    """Parses a size string in the format of "{width}x{height}" and returns a tuple of integers (width, height).



    Args:

        size_str: The size string to parse.



    Raises:

        ValueError: If the size string is in an invalid format.

    """

    try:

        width, height = size_str.split("x")

        return int(width), int(height)

    except ValueError:

        raise ValueError("Invalid size string format.")

