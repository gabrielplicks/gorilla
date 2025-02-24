from typing import Tuple



def find_first_substring(string: str, substring: str) -> int:

    """Finds the first occurrence of a substring within a string.



    Args:

        string: The string to search within.

        substring: The substring to search for.



    Returns:

        The index of the first occurrence of the substring, or -1 if the substring is not found.

    """

    if not string or not substring or len(substring) > len(string):

        return -1



    for i in range(len(string)):

        if string[i] == substring[0]:

            for j in range(len(substring)):

                if i + j >= len(string) or string[i + j] != substring[j]:

                    break

            else:

                return i

    return -1

