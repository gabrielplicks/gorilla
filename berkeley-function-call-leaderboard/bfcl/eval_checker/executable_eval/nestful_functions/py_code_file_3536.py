import copy



def copy_list_shallow(nested_list: list[list[int]]) -> list[list[int]]:

    """Creates a shallow copy of a list of lists while sharing the same sublists in memory.



    Args:

        nested_list: A list of lists of integers.



    Returns:

        A shallow copy of the input list.

    """

    return copy.copy(nested_list)

