from typing import List



def lower_first_item(*args: List[List[str]]) -> List[str]:

    """Lowercases the first item of each argument list.



    Args:

        args: One or more lists of strings.



    Returns:

        A new list with the first item of each argument list lowercased.

    """

    return [lst[0].lower() for lst in args if lst]

