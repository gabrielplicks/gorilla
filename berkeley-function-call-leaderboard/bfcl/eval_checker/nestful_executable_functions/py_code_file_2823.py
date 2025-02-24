from typing import List, Union



def flatten_non_negative_integers(nested_list: List[Union[int, List[Union[int, float, str]]]]) -> List[int]:

    """Flattens a nested list of integers and returns a list containing only the non-negative integers.



    Args:

        nested_list: A nested list of integers and other data types.



    Returns:

        A flattened list containing only the non-negative integers.

    """

    flattened_list = []



    def helper(elem: Union[int, List[Union[int, float, str]]]):

        if isinstance(elem, int) and elem >= 0:

            flattened_list.append(elem)

        elif isinstance(elem, list):

            for item in elem:

                helper(item)



    for item in nested_list:

        helper(item)



    return flattened_list

