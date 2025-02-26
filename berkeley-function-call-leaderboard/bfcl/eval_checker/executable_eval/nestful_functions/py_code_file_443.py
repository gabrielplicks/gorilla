from typing import Union



def add_paddings(text: str, padding_width: Union[int, float]) -> str:

    """Adds equal padding to both sides of a string.



    Args:

        text: The input string.

        padding_width: The width of the padding. Must be a positive integer.



    Raises:

        TypeError: If padding_width is not an integer or is less than 0.

    """

    if not isinstance(padding_width, int) or padding_width < 0:

        raise TypeError('padding_width must be a positive integer')



    if padding_width == 0:

        return text



    padding = ' ' * padding_width

    return padding + text + padding

