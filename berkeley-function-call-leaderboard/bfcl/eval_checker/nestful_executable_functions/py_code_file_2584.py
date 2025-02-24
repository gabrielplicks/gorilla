from typing import List, Dict



def func_refactor(strings: List[str]) -> (List[str], Dict[str, int]):

    """

    This function takes a list of strings and returns a tuple of two elements.

    The first element is a list of all strings that contain the substring '777'.

    The second element is a dictionary where the keys are the strings that contain the substring '777' and the values are the lengths of the strings.



    Args:

        strings: A list of strings.



    Returns:

        A tuple containing a list of strings and a dictionary.

    """

    new_list = [s for s in strings if '777' in s]

    new_dict = {s: len(s) for s in strings if '777' in s}

    return new_list, new_dict

