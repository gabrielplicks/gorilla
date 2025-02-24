from typing import Dict, List, Any



def is_ancestor(tree: Dict[str, List[str]], ancestor: str, descendant: str) -> bool:

    """Determines whether a given node is an ancestor of another node in a tree represented as a dictionary.



    Args:

        tree: A dictionary representing the tree.

        ancestor: The name of the node to check if it is an ancestor.

        descendant: The name of the node to check if it is a descendant.



    Returns:

        True if the first node is an ancestor of the second node, False otherwise.

    """

    if ancestor == descendant:

        return True



    children = tree.get(ancestor)



    if children is None:

        return False



    return any(is_ancestor(tree, child, descendant) for child in children)

