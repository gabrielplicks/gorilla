from typing import List, Union



def quote_string(s: str) -> str:

    """Encloses a string in double quotes.



    Args:

        s: The string to enclose in double quotes.



    Returns:

        The string enclosed in double quotes.

    """

    return f'"{s}"'



def list_to_csv(nested_list: List[List[str]]) -> str:

    """Converts a nested list of strings into a single string containing the concatenated elements, where each element is separated by a comma and enclosed in double quotes.



    Args:

        nested_list: The nested list of strings to convert.



    Returns:

        The concatenated string.

    """

    formatted_strings = []



    for sublist in nested_list:

        for element in sublist:

            formatted_strings.append(quote_string(element))



    return ", ".join(formatted_strings)

