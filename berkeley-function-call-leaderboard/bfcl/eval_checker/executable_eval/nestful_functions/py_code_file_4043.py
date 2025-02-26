from typing import Any, Dict



def extract_tree_content(tree: Dict[str, Any]) -> list:

    """Extracts the content of a tree in a list format using preorder traversal.



    Args:

        tree: A dictionary representing the tree structure.



    Returns:

        A list containing the content of the tree in preorder traversal.

    """

    content = []



    def traverse(node: Dict[str, Any]):

        if node:

            content.append(node["data"])

            traverse(node["left"])

            traverse(node["right"])



    traverse(tree)

    return content

