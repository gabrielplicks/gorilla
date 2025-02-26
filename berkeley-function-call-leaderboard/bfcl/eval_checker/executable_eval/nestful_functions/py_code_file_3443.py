from typing import List



def all_rotations(string: str) -> List[str]:

    """

    Returns all possible rotations of a given string.



    Args:

        string: The input string.



    Returns:

        A list of rotated strings.

    """

    rotated_strings = []



    for i in range(len(string)):

        rotated_string = string[i:] + string[:i]

        rotated_strings.append(rotated_string)



    return rotated_strings

