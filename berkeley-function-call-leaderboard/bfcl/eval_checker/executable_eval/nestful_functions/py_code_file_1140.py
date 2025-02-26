from typing import List



def filter_matching_strings(str_list: List[str], str_list_list: List[List[str]]) -> List[str]:

    """Filters a list of strings based on a list of lists of strings.

    Args:

        str_list: The list of strings to filter.

        str_list_list: The list of lists of strings to match against.

    """

    filtered_strings = []

    for str1 in str_list:

        for str2 in str_list_list:

            if all(elem in str1 for elem in str2):

                filtered_strings.append(str1)



    return filtered_strings

