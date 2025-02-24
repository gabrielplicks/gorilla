from typing import List



def exclude_values(include_list: List[str], exclude_list: List[str]) -> List[str]:

    """

    Returns a new list containing only the values that are present in `include_list` and not present in `exclude_list`.



    Args:

        include_list: The list of values to include.

        exclude_list: The list of values to exclude.

    """

    result = []

    exclude = set(exclude_list)

    for item in include_list:

        if item not in exclude:

            result.append(item)

    return result

