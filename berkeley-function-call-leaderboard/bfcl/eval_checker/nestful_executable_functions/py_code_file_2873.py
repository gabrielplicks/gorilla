from typing import List



def get_setup_string(string: str) -> str:

    """Transforms a string to a setup string, which is a string with its characters in alphabetical order.



    Args:

        string: The input string.



    Returns:

        The setup string.

    """

    char_list: List[str] = list(string)

    char_list.sort()

    return "".join(char_list)

