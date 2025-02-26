from typing import Optional



class TreeNode:

    def __init__(self, value: int):

        self.value = value

        self.left: Optional[TreeNode] = None

        self.right: Optional[TreeNode] = None



def find_height(root: Optional[TreeNode]) -> int:

    """Finds the height of a binary tree.



    The height of a binary tree is defined as the number of nodes from the root node to the deepest leaf node.



    Args:

        root: The root node of the binary tree.



    Returns:

        The height of the binary tree.

    """

    if root is None:

        return 0

    return max(find_height(root.left), find_height(root.right)) + 1

