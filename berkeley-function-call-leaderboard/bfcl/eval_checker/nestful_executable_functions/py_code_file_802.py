from typing import List



def print_list(list: List[str]) -> str:

    """Prints the elements of a list in the following format:

    "1", "2", "3", ..., "N"

    If the list is empty, returns an empty string. The list contains only strings.

    Args:

        list: The list of strings to print.

    """

    if len(list) == 0:

        return ""



    result = []

    for element in list:

        result.append(f'"{element}"')



    return ", ".join(result)

