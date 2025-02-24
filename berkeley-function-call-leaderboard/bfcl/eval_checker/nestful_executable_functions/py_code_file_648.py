from typing import List



def find_first_index_of_second_array(arr1: List[int], arr2: List[int]) -> int:

    """Finds the first index of the second array in the first array.



    Args:

        arr1: The first array of bytes.

        arr2: The second array of bytes.



    Returns:

        The index if it exists or -1 if the second array is not found in the first array.

    """

    n1 = len(arr1)

    n2 = len(arr2)

    for i in range(n1 - n2 + 1):

        if arr1[i] == arr2[0] and arr1[i + 1:i + n2] == arr2[1:]:

            return i

    return -1

