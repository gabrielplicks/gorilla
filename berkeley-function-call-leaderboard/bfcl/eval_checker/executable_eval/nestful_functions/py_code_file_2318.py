from typing import List



def get_first_10_lines(filename: str) -> List[str]:

    """Returns the first 10 lines of the file in a list of strings.

    If the file does not exist, returns an empty list.

    If the file contains less than 10 lines, returns all the lines in the file.



    Args:

        filename: The name of the file.

    """

    try:

        with open(filename, "r") as file:

            lines = file.readlines()

    except FileNotFoundError:

        return []



    if len(lines) > 10:

        return lines[:10]

    else:

        return lines

