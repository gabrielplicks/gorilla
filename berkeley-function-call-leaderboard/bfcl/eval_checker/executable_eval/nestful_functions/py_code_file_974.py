from typing import Optional



class Node:

    def __init__(self, value: int):

        self.value = value

        self.left: Optional[Node] = None

        self.right: Optional[Node] = None



def find_maximum(root: Optional[Node]) -> int:

    """Finds the maximum value in a binary tree.

    Args:

        root: The root node of the binary tree.

    """

    if root is None:

        return float('-inf')

    left_max = find_maximum(root.left)

    right_max = find_maximum(root.right)

    return max(root.value, left_max, right_max)

