from typing import List



def format_nested_list(nested_list: List[List[str]]) -> str:

    """Formats a list of lists of strings into a single string, where each element is separated by a space, and each row is separated by a newline.



    Args:

        nested_list: A list of lists of strings.



    Returns:

        A formatted string.

    """

    formatted_list = []

    for row in nested_list:

        row_string = ' '.join(row)

        formatted_list.append(row_string)



    return '\n'.join(formatted_list)

