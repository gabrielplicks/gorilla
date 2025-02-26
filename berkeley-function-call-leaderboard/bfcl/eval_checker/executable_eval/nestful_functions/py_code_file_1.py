from typing import List



def find_strings_with_pattern(string_list: List[str], pattern: str) -> List[int]:

    """Finds the indices of strings in a list that contain a given substring.



    Args:

        string_list: A list of strings to search for the pattern.

        pattern: The substring pattern to search for in the strings.



    Returns:

        A list of indices of the strings in the input list that contain the pattern as a substring.

    """

    result = []



    for i, string in enumerate(string_list):

        if pattern in string:

            result.append(i)



    return result

