from typing import List



def sort_by_first_digit(string_list: List[str]) -> None:

    """Sorts a list of strings based on the first digit in each string.



    If the first digit of a string is not a valid integer, it is treated as if it were 0.

    If the first digit of multiple strings is the same, the strings are sorted alphabetically.



    Args:

        string_list: A list of strings to be sorted.

    """

    def get_first_digit_value(string: str) -> int:

        if string[0].isdigit():

            return int(string[0])

        else:

            return 0



    string_list.sort(key=lambda x: get_first_digit_value(x))

