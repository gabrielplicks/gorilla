import re

from typing import Union



def normalize_string(s: Union[str, bytes]) -> str:

    """Normalizes a string by replacing all non-alphanumeric characters with underscores and converting to lowercase.

    The function also groups repeated underscores and trims leading and trailing underscores.

    Args:

        s: The input string to be normalized.

    """

    normalized_string = re.sub(r"\W", "_", s)

    normalized_string = re.sub(r"__+", "_", normalized_string)

    normalized_string = normalized_string.strip("_")

    return normalized_string.lower()

