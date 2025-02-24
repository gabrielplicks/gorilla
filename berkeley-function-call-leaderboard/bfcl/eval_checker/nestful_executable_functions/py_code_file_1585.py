from typing import List

import string



def generate_string_table() -> List[List[str | int]]:

    """Generates a string table that consists of all ASCII characters and their corresponding numerical values.

    The string table is a two-dimensional array with 256 rows and 2 columns.

    The first column is the character, and the second column is its numerical value.



    Returns:

        A list of lists, where each inner list contains a character and its numerical value.

    """

    string_table = []

    for char in string.printable:

        string_table.append([char, ord(char)])

    return string_table

