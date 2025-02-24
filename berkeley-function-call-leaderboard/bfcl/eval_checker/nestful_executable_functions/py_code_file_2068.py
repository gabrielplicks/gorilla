from typing import Tuple



def is_same_byte_array(first_array: bytes, second_array: bytes) -> bool:

    """Checks if two byte arrays are equal in length and contain the same elements.



    Args:

        first_array: The first byte array.

        second_array: The second byte array.



    Returns:

        True if the arrays are equal in length and contain the same elements, False otherwise.

    """

    if len(first_array) != len(second_array):

        return False



    for i in range(len(first_array)):

        if first_array[i] != second_array[i]:

            return False



    return True

