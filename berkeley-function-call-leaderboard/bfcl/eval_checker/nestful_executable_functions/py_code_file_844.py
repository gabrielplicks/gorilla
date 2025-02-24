from typing import List, Union



def sum_int(input_list: List[Union[int, str]]) -> Union[int, None]:

    """Calculates the sum of integers in a list.



    Args:

        input_list: A list of numbers.



    Returns:

        The sum of integers in the list, or None if the list contains any non-integers.

    """

    if not isinstance(input_list, list):

        return 0



    total = 0

    for item in input_list:

        if not isinstance(item, int):

            return None

        total += item

    return total

