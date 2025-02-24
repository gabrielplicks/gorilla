from typing import List, Tuple



def flatten_and_count_values(nested_list: List[List[int]]) -> List[Tuple[int, int]]:

    """Flattens a list of lists and counts the occurrences of each value.



    Args:

        nested_list: A list of lists containing integers.



    Returns:

        A list of tuples, where each tuple consists of a value and its count.

    """

    flattened_list = set([item for sublist in nested_list for item in sublist])

    count_dict = {}

    for item in flattened_list:

        count_dict[item] = sum(item in sublist for sublist in nested_list)



    return [(item, count_dict[item]) for item in flattened_list]

