from typing import List



def my_split(line: str, sep: str = None) -> List[str]:

    """Splits a string based on a separator.



    Args:

        line: The string to split.

        sep: The separator to split the string on. If not specified, the string is split on whitespace. If an empty string, the string is split on every character.



    Returns:

        A list of strings.

    """

    if sep is None:

        return line.split()

    elif sep == "":

        return [char for char in line]

    else:

        return line.split(sep)

