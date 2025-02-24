from typing import Tuple



def get_parent_child(path: str) -> Tuple[str, str]:

    """Returns a tuple of the parent node and the child node in a string with forward slash separators.



    Args:

        path: The input string with forward slash separators.



    Returns:

        A tuple of the parent node and the child node.

    """

    nodes = path.split("/")

    if len(nodes) == 1:

        return ("", nodes[0])

    else:

        return (nodes[-2], nodes[-1])

