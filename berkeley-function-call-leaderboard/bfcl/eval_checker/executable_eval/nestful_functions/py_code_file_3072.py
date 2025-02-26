from typing import List



def combine_and_format_string_list(str_list: List[str]) -> str:

    """Combines a list of strings into a single string, separated by commas and with each word surrounded by double quotes.

    If the list is empty, returns an empty string.



    Args:

        str_list: A list of strings.



    Returns:

        A string with the combined and formatted list of strings.

    """

    if len(str_list) == 0:

        return ""



    formatted_list = []



    for string in str_list:

        formatted_string = f'"{string}"'

        formatted_list.append(formatted_string)



    combined_string = ",".join(formatted_list)

    return combined_string

