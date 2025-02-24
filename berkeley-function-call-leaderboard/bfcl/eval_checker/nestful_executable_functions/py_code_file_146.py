from typing import List, Any



def check_array_length(array: Any) -> bool:

    """Checks the length of the given array and returns True if the length is 0, 1, or 2, and returns False otherwise.

    If the input is not an array, it should return False and print an error message.



    Args:

        array: The input array.



    Returns:

        True if the length of the array is 0, 1, or 2, False otherwise.

    """

    if not isinstance(array, List):

        print("Error: Input is not an array.")

        return False

    else:

        return len(array) in [0, 1, 2]

