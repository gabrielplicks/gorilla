import re

import typing



def words_from_string(string: str) -> typing.List[str]:

    """Returns a list of all the words contained in the string.



    Args:

        string: The input string containing words and whitespace characters.



    Returns:

        A list of words.

    """

    return re.findall(r"\w+", string)

