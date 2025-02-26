from typing import Optional



class TreeNode:

    def __init__(self, val=0, left=None, right=None):

        self.val = val

        self.left = left

        self.right = right



size: int = 0  # Global variable to keep track of the size of the tree



def count_nodes_recursive(node: Optional[TreeNode]) -> int:

    """Counts the total number of nodes in a binary search tree using a recursive approach.



    Args:

        node: The root node of the binary search tree.



    Returns:

        The total number of nodes in the binary search tree.

    """

    if node is None:

        return 0

    size = count_nodes_recursive(node.left) + count_nodes_recursive(node.right) + 1

    return size

