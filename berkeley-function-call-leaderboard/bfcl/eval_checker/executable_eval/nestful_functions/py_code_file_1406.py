from typing import List



def format_list_with_brackets(strs: List[str]) -> str:

    """Formats a list of strings into a single string with brackets and commas.



    Args:

        strs: A list of strings.



    Returns:

        A formatted string with brackets and commas.

    """

    if not strs:

        return ''



    formatted_string = '('

    for i, str in enumerate(strs):

        if i > 0:

            formatted_string += ', '

        formatted_string += str

    formatted_string += ')'



    return formatted_string

