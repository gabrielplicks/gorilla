from typing import List



def pad_buffer(lst: List[int], n: int) -> List[int]:

    """Pads a list of integers with zeros until its length is a multiple of a given integer.



    Args:

        lst: The list of integers to pad.

        n: The integer to divide the length of the list by.

    """

    lst.reverse()  # Reverse the list

    lst_len = len(lst)

    if lst_len % n != 0:  # Check if the length is not a multiple of n

        lst.extend([0] * (n - (lst_len % n)))  # Append the required number of zeros

    return lst

