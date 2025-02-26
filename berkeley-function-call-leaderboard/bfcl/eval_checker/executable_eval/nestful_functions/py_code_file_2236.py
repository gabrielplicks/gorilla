from typing import Union



def convert_to_int_or_none(value: Union[int, float, None]) -> Union[int, None]:

    """Converts a value to an integer or returns None if the value is not an integer or is None.

    Args:

        value: The value to be converted.

    """

    if value is None:

        return None

    if isinstance(value, int):

        return value

    if isinstance(value, float):

        return int(value)

    return None

