from typing import List



def median_of_list_with_error_handling(lst: List[int]) -> Union[float, str]:

    """Calculates the median of a list of integers.

    Args:

        lst: A list of integers.

    Returns:

        The median of the list, or a string indicating an error if the list is empty or contains negative integers.

    """

    if not lst or any(x < 0 for x in lst):

        return "Invalid input: the list cannot be empty or contain negative integers"



    sorted_lst = sorted(lst)

    n = len(lst)



    if n % 2 == 1:

        return sorted_lst[n // 2]

    else:

        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2

