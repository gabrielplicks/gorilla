from typing import List



def new_list_of_lists(list_of_lists: List[List[int]], start_index: int) -> List[List[int]]:

    """Creates a new list of lists from the given list of lists, starting from the specified index.

    Args:

        list_of_lists: The original list of lists.

        start_index: The index to start from. If negative, start from the end of the list.

    """

    return [sublist[start_index:] if start_index >= 0 else sublist[-start_index:] for sublist in list_of_lists]

