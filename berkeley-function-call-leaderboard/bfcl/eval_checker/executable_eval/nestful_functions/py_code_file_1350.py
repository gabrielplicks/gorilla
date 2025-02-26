from typing import Union



def find_pattern_index(s: Union[str, None], p: Union[str, None]) -> int:

    """Finds the index of the first occurrence of the pattern `p` within the string `s`, ignoring case.



    Args:

        s: The input string.

        p: The pattern to search for.



    Returns:

        The index of the first occurrence of the pattern `p` within the string `s`, ignoring case.

        If the pattern is not found, returns `-1`.



    Raises:

        ValueError: If the input parameters are not strings.

    """

    if not isinstance(s, str) or not isinstance(p, str):

        raise ValueError("Input parameters must be strings")



    s_lower = s.lower()

    p_lower = p.lower()



    index = s_lower.find(p_lower)

    return index

