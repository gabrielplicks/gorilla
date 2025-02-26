from typing import Dict, List, Tuple



def count_unique_ints(input_list: List[int]) -> List[int]:

    """Calculates the count of the number of occurrences of each unique integer in the input list, sorted in ascending order of the integers themselves.



    Args:

        input_list: A list of integers.



    Returns:

        A list of integers that represent the count of the number of occurrences of each unique integer in the input list, sorted in ascending order of the integers themselves.

    """

    count_dict: Dict[int, int] = {}

    for num in input_list:

        if num not in count_dict:

            count_dict[num] = 0

        count_dict[num] += 1



    return sorted(count_dict.items(), key=lambda x: x[0])

