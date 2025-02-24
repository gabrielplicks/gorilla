from typing import Any, Dict, List



def find_all_paths(dictionary: Dict[Any, Any]) -> List[List[Any]]:

    """Finds all paths from the root to leaf nodes in a nested dictionary.



    A path is a list of keys, and a leaf node is a node without any children.

    The function handles the case where the dictionary may contain cycles,

    so the paths returned may not be unique.



    Args:

        dictionary: The nested dictionary to traverse.



    Returns:

        A list of all paths from the root to leaf nodes.

    """

    paths = []



    def traverse(node: Any, path: List[Any]) -> None:

        if not isinstance(node, dict):

            paths.append(path)

        else:

            for key, value in node.items():

                traverse(value, path + [key])



    traverse(dictionary, [])

    return paths

