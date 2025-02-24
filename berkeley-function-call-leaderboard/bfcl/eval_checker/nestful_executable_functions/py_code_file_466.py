from typing import List



def bytearray_to_string(arr: List[int]) -> str:

    """Converts a bytearray object to a string representation with each byte separated by commas.



    Args:

        arr: The bytearray object to convert.



    Returns:

        The string representation of the bytearray.

    """

    string = ""

    for i in range(len(arr)):

        string += str(arr[i])

        if i < len(arr) - 1:

            string += ","

    return string

