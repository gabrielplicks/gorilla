from typing import Optional



class Node:

    def __init__(self, value):

        self.value = value

        self.left = None

        self.right = None



    def is_leaf(self):

        return self.left is None and self.right is None



def max_depth_binary_tree(root: Optional[Node]) -> int:

    """Calculates the maximum depth of a binary tree.

    Args:

        root: The root node of the binary tree.

    """

    if root is None:

        return 0

    if root.is_leaf():

        return 1

    return 1 + max(max_depth_binary_tree(root.left), max_depth_binary_tree(root.right))

