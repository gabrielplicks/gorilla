from typing import List, Set



def compute_intersection(input_str: str, input_list: List[int]) -> List[int]:

    """Computes the intersection of two sets of numbers.



    The first set is in a string format, so it is converted into a list of integers.

    Then, set intersection is performed using the built-in `set.intersection()` method.

    The resulting set is converted into a list using list comprehension.



    Args:

        input_str: The first set of numbers in a string format.

        input_list: The second set of numbers in a list format.



    Returns:

        A list of integers representing the intersection of the two sets.

    """

    input_set = set(int(num) for num in input_str.split(","))

    intersection_set = input_set.intersection(set(input_list))

    return [num for num in intersection_set]

