from typing import List



def string_list_to_int_list(string_list: List[str]) -> List[int]:

    """Converts a list of strings containing only digits into a list of integers.



    Args:

        string_list: A list of strings containing only digits.



    Raises:

        ValueError: If the string list contains empty strings or strings containing non-digit characters.

    """

    int_list = []

    for string in string_list:

        if string.isdigit():

            int_list.append(int(string))

        else:

            raise ValueError("String list contains non-digit elements.")

    return int_list

