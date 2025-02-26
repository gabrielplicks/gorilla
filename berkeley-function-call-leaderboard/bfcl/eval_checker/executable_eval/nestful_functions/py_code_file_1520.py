from typing import List



def select_nth_largest(arr: List[int], n: int) -> int:

    """Selects the Nth largest element in the array.

    Args:

        arr: A list of integers.

        n: The Nth largest element to select.

    """

    arr = sorted(arr, key=lambda x: -abs(x))  # Sort the array in descending order by absolute value

    return arr[n-1]

