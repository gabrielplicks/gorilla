from typing import Union



def format_scalar(scalar: Union[float, int]) -> str:

    """Converts a scalar into a string with a particular format.

    Args:

        scalar: The scalar to be formatted.

    """

    if isinstance(scalar, float):

        formatted_value = str(round(scalar)) + 'f'

    elif isinstance(scalar, int):

        formatted_value = str(scalar) + 'i'

    return formatted_value

