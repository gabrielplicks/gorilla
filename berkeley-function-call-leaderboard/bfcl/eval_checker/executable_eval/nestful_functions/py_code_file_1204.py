from typing import Union



def convert_to_int_or_default(value: Union[str, int, float], default_value: Union[int, float] = 0) -> int:

    """Converts a value to an integer or returns a default value if the value cannot be converted.



    Args:

        value: The value to be converted to an integer.

        default_value: The default value to be returned if the conversion fails.

    """

    try:

        return int(value)

    except ValueError:

        return default_value

