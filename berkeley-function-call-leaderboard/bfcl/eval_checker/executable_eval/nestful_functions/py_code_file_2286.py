from typing import List



def convert_to_format(input_list: List[str]) -> str:

    """Converts a list of strings to a single string, where each string is formatted as follows:

    `"{1}: {2}"`. The first string should be the string representation of the list index,

    the second should be the original list string, and there should be no comma or space

    between the two. For example, the input list ["a", "b", "c"] should result in the output

    string "0: a1: b2: c".



    Args:

        input_list: A list of strings to be converted.



    Returns:

        A single string formatted as described above.

    """

    return ' '.join([f'{i}: {s}' for i, s in enumerate(input_list)])

