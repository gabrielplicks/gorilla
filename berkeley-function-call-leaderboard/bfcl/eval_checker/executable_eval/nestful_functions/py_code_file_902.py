from typing import List



def concatenate_with_separator(list_of_strings: List[str]) -> str:

    """Concatenates the elements of a list into a single string separated by a comma and a space.



    Args:

        list_of_strings: A list of strings to concatenate.



    Returns:

        A single string that concatenates the elements of the input list, with each element separated by a comma and a space.

    """

    result = ''



    for i, element in enumerate(list_of_strings):

        if i == len(list_of_strings) - 1:

            result += element

        else:

            result += element + ', '



    return result

