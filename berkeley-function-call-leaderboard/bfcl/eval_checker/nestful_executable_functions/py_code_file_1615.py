from typing import List



def remove_dups(input_list: List[int]) -> List[int]:

    """

    Removes duplicates from a list of integers and returns a new sorted list.



    Args:

        input_list: A list of integers.



    Returns:

        A new sorted list of unique integers.

    """

    sorted_list = sorted(input_list)

    unique_values = set(sorted_list)

    deduplicated_list = sorted(unique_values)

    return deduplicated_list

