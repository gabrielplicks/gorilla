from typing import List



def concatenate_n_to_string(string: str, n: int) -> str:

    """Concatenates the string representation of n to the end of the input string.



    Args:

        string: The input string.

        n: The number to be concatenated to the end of the string.

    """

    return "{}{}".format(string, n)



def add_n_to_strings(strings: List[str], n: int) -> List[str]:

    """Concatenates the string representation of n to the end of each string in the input list.



    Args:

        strings: The input list of strings.

        n: The number to be concatenated to the end of each string.

    """

    return list(map(lambda string: concatenate_n_to_string(string, n), strings))

