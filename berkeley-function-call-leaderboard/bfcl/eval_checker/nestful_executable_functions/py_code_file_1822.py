from typing import List



def find_containing_strings(string_list: List[str], target_string: str) -> List[str]:

    """Finds all strings in the list that contain the target string.



    Args:

        string_list: A list of strings.

        target_string: The target string to search for.



    Returns:

        A list of strings that contain the target string.

    """

    result = []

    for string in string_list:

        if target_string in string:

            result.append(string)

    return result

