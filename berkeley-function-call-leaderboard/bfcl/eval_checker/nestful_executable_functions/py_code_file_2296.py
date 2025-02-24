from typing import List



def convert_to_lengths(arr: List[str]) -> List[int]:

    """Converts a list of strings to a list of their lengths.



    Args:

        arr: A list of strings.



    Raises:

        AssertionError: If the input is not a list of strings.

    """

    assert isinstance(arr, list) and all(isinstance(item, str) for item in arr), "Input must be a list of strings"

    return [len(item) for item in arr]

