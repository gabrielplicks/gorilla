from typing import List



def concatenate_with_commas(my_list: List[str]) -> str:

    """Concatenates all strings in a list into a single string, separated by commas and spaces.



    Args:

        my_list: A list of strings to be concatenated.



    Returns:

        A string with all the input strings concatenated, separated by commas and spaces.

    """

    result = ""

    for item in my_list:

        result += item + ", "

    result = result[:-2]  # Remove the last comma and space



    return result

