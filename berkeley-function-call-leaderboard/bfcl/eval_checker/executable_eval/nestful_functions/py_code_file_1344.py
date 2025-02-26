from typing import Union



def calculate_area(width: Union[int, None], length: Union[int, None]) -> int:

    """Calculates the area based on the width and length.

    Args:

        width: The width of the area.

        length: The length of the area.

    """

    if width is None and length is None:

        return 0

    if not isinstance(width, int) or not isinstance(length, int):

        raise Exception('One or more arguments are not integers')

    if width < 0 or length < 0:

        raise Exception('One or more arguments are negative')

    return width * length

