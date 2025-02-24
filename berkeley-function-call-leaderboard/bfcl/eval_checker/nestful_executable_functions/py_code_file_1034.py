import sys



def get_byte_size(int_list: List[int]) -> int:

    """Calculates the total number of bytes needed to store a list of integers as a byte array.



    Args:

        int_list: A list of integers.



    Returns:

        The total number of bytes needed to store the list and its integers.

    """

    size = sys.getsizeof(int_list)  # size of list itself

    size += len(int_list) * sys.getsizeof(int_list[0])  # size of each integer

    return size

