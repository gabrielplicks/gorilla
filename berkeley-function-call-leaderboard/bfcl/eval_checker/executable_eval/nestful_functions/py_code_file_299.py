from typing import Optional



class Node:

    def __init__(self, value: int):

        self.value = value

        self.left: Optional[Node] = None

        self.right: Optional[Node] = None



def count_size(root: Optional[Node]) -> int:

    """Counts the number of nodes in a binary tree.

    Args:

        root: The root node of the binary tree.

    Returns:

        The number of nodes in the binary tree.

    """

    if root is None:

        return 0

    left_count = count_size(root.left)

    right_count = count_size(root.right)

    return 1 + left_count + right_count

