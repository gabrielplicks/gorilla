from typing import List, Union



def topological_sort(data: List[Union[int, List[Union[int, List[int]]]]]) -> List[int]:

    """Performs a topological sort traversal on a nested list data structure.



    The function takes a nested list data structure as input and returns a list of the nodes in the order they are encountered during the traversal.



    Args:

        data: The nested list data structure to traverse.



    Returns:

        A list of the nodes in the order they are encountered during the traversal.

    """

    extracted_nodes = []

    def traverse(node):

        if isinstance(node, list):

            for element in node:

                traverse(element)

        else:

            extracted_nodes.append(node)

    traverse(data)

    return extracted_nodes

