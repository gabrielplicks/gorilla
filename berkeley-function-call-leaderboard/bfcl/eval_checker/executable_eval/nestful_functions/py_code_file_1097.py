from typing import List, Callable



def lower_case(s: str) -> str:

    """Converts a string to lower case.



    Args:

        s: The input string.



    Returns:

        The lower-cased version of the input string.

    """

    return s.lower()



def process_list(lst: List[str], process_fn: Callable[[str], str]) -> List[str]:

    """Applies a function to each element in a list of strings.



    Args:

        lst: The input list of strings.

        process_fn: The function to apply to each element in the list.



    Returns:

        A list of processed strings.

    """

    return list(map(process_fn, lst))

