from typing import List, Union



def flat_list(lst: List[Union[int, str, List[Union[int, str]]]]) -> List[Union[int, str]]:

    """Flattens a nested list of integers or strings into a 1-dimensional list.

    Args:

        lst: The nested list to be flattened.

    """

    if not lst:

        return []

    head, *tail = lst

    if isinstance(head, list):

        return flat_list(head) + flat_list(tail)

    return [head] + flat_list(tail)

