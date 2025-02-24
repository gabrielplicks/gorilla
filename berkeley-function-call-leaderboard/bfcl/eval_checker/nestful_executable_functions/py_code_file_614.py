from string import ascii_uppercase

from typing import Union



def get_excel_header(index: Union[int, float]) -> str:

    """

    Returns the Excel column header corresponding to the given index.



    Args:

        index: The index of the column header.



    Returns:

        The column header as a string.

    """

    index -= 1

    header = ""

    while index >= 0:

        remainder = index % 26

        letter = ascii_uppercase[remainder]

        header = letter + header

        index = index // 26 - 1

    return header

