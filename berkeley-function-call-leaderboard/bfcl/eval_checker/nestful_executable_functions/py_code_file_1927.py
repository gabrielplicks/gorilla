from typing import List



def find_xyz_indices(string: str) -> List[int]:

    """Finds the indices of all occurrences of the pattern 'xyz' within the string.



    Args:

        string: The input string.



    Returns:

        A list of indices where the pattern 'xyz' is found.

    """

    result = []

    window_size = 3

    for i in range(len(string) - window_size + 1):

        window = string[i:i+window_size]

        if window == "xyz":

            result.append(i)

    return result

