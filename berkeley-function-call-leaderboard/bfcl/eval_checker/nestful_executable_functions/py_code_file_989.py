from typing import Tuple



def calculate_parent_node(node_index: int, total_nodes: int) -> int:

    """Calculates the parent node of a node in a binary heap.



    Args:

        node_index: The index of the node.

        total_nodes: The total number of nodes in the heap.



    Returns:

        The index of the parent node.



    Raises:

        IndexError: If the parent node's index is not within the range of valid indices for the heap.

    """

    parent_index = (node_index - 1) // 2



    if parent_index < 0 or parent_index >= total_nodes:

        raise IndexError("Invalid node or parent index")



    return parent_index

