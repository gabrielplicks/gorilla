from typing import List, Optional



class Node:

    def __init__(self, value: str):

        """

        Node class for a binary tree.

        Args:

            value: The value of the node.

        """

        self.value = value

        self.children = []



def traverse_binary_tree(root_node: Optional[Node]) -> List[str]:

    """

    Traverses a binary tree in breadth-first order, starting at the root node.

    Args:

        root_node: The root node of the binary tree.

    Returns:

        A list of node values in breadth-first order.

    """

    if not root_node:

        return []

    queue = [root_node]

    results = []

    while queue:

        node = queue.pop(0)

        results.append(node.value)

        queue.extend(node.children)

    return results

