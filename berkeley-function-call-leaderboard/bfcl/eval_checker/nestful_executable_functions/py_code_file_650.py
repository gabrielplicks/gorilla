from typing import List



def array_to_string(arr: List[int]) -> str:

    """Converts an array of integers into a string with the format "[1, 2, 3]".

    Args:

        arr: The input array of integers.

    """

    formatted_arr = [str(element) for element in arr]

    return f"[{', '.join(formatted_arr)}]"

