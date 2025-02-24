from typing import Any, List



def flatten_helper(item: Any, result: List[Any]) -> None:

    """Recursively flattens a list of nested lists.



    Args:

        item: The item to be flattened.

        result: The result list to store the flattened elements.

    """

    if isinstance(item, list):

        for sub_item in item:

            flatten_helper(sub_item, result)

    else:

        result.append(item)



def flatten(data: List[Any]) -> List[Any]:

    """Flattens a list of nested lists.



    Args:

        data: The input list of nested lists.



    Returns:

        The flattened list.

    """

    result = []

    for item in data:

        flatten_helper(item, result)

    return result

