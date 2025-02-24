from typing import List



def str_list_join(strs: List[str], sep: str) -> str:

    """Joins a list of strings `strs` with a specified separator string `sep`.



    Args:

        strs: A list of strings to join.

        sep: A separator string.

    """

    return sep.join(strs)

