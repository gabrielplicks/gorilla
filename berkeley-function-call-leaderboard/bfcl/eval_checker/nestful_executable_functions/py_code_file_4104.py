from typing import List



def sum_next(arr: List[int]) -> List[int]:

    """Calculates the sum of each element in the input array and the next element in the array.



    Args:

        arr: The input array.



    Returns:

        A list of the same size as the input array, where each element is the sum of the corresponding element in `arr` and the next element in `arr`.

    """

    return [arr[i] + arr[i+1] if i < len(arr)-1 else arr[i] for i in range(len(arr))]

