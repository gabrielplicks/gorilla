from typing import List



def string_splitter(my_string: str) -> List[str]:

    """Splits a string into substrings based on the "|" character and returns a list of strings.

    If the "|" character is not present, the function inserts the entire string into the list.

    Args:

        my_string: The string to be split.

    """

    splitted_string = my_string.split("|")

    if len(splitted_string) > 1:

        return splitted_string

    else:

        return [my_string]

