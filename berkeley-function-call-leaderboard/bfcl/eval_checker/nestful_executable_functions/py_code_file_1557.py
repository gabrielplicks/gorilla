import bisect



def insert_sorted(arr: list, val: int) -> list:

    """Inserts a value into a sorted array at a position such that the array is still sorted.



    Args:

        arr: The array to be inserted into.

        val: The value to be inserted.



    Returns:

        The array after the insertion.

    """

    pos = bisect.bisect_left(arr, val)

    arr.insert(pos, val)

    return arr

