from typing import List



def first_characters(string_list: List[str]) -> List[str]:

    """Creates a new list that contains the first character of each string in the input list.



    Args:

        string_list: A list of strings.



    Returns:

        A list of the first characters of the input strings.

    """

    return [s[0] for s in string_list]

