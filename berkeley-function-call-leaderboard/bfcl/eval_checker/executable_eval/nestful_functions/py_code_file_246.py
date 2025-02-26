from typing import Optional



class TreeNode:

    def __init__(self, data: int):

        self.data = data

        self.left: Optional[TreeNode] = None

        self.right: Optional[TreeNode] = None



def invert_binary_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:

    """Inverts a binary tree by swapping each node's left and right child nodes.



    Args:

        root: The root node of the binary tree.



    Returns:

        The root node of the inverted binary tree.

    """

    if not root:

        return None

    new_root = TreeNode(root.data)

    new_root.left = invert_binary_tree(root.right)

    new_root.right = invert_binary_tree(root.left)

    return new_root

