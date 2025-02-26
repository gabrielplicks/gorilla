from typing import Union



def convert_to_positive_integer(value: Union[int, float]) -> int:

    """Converts a value to a positive integer.

    Args:

        value: The value to convert.

    Raises:

        ValueError: If the value is not a positive integer.

    """

    if not isinstance(value, (int, float)):

        raise ValueError('Not a positive integer')

    if value < 0:

        raise ValueError('Not a positive integer')

    return int(value)

