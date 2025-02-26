from typing import List



def angle_rotation(arr: List[int], k: int) -> List[int]:

    """Rotates the elements in the array by `k` positions in the right direction.



    Args:

        arr: The array of numbers.

        k: The number of positions to rotate the elements in the array.



    Returns:

        The array after rotating the elements by `k` positions in the right direction.

    """

    if len(arr) < k:

        k = k % len(arr)

    return arr[len(arr)-k:] + arr[:len(arr)-k]

