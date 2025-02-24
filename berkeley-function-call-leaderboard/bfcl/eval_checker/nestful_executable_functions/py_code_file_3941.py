from typing import List, Dict



def get_sum_of_foo(obj_list: List[Dict[str, int]]) -> int:

    """Calculates the sum of the `foo` attribute of all objects in a list.



    Args:

        obj_list: A list of objects, where each object is a dictionary with a `foo` attribute.



    Returns:

        The sum of the `foo` attribute of all objects in the list.

    """

    return sum(obj['foo'] for obj in obj_list)

