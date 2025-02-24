from typing import List, Union



def invert_dot_separated_string(s: str) -> str:

    """Inverts the order of a string with dot-separated path components.



    Args:

        s: A string with dot-separated path components.



    Returns:

        A string with the order of the path components inverted.

    """

    components: List[str] = s.split(".")

    inverted_components: List[str] = components[::-1]

    inverted_string: str = ".".join(inverted_components)

    return inverted_string

