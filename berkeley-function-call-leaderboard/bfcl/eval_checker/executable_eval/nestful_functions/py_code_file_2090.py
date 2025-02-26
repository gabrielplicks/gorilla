from typing import List



def prefix_checker(input_string: str, string_list: List[str]) -> bool:

    """Determines if the input string is a prefix of any of the strings in the list.



    Args:

        input_string: The input string to check.

        string_list: The list of strings to check against.



    Returns:

        True if the input string is a prefix of any of the strings in the list, False otherwise.

    """

    for string in string_list:

        if string.startswith(input_string):

            return True

    return False



input_string = "ABC"

string_list = ["ABCDE", "ABCD", "ABC"]

result = prefix_checker(input_string, string_list)

print(result)

