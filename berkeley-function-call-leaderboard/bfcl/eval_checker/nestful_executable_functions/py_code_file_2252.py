from typing import Dict, List, Any



def is_leaf_node(node_id: int, nodes: List[Dict[str, Any]]) -> bool:

    """Determines if a node is a leaf node (no children) in a list of nodes.



    Args:

        node_id: The ID of the node to check.

        nodes: A list of nodes, each of which is a dictionary with the keys 'id', 'parent_id', and 'children'.



    Returns:

        True if the node is a leaf node, False otherwise.

    """

    nodes_dict = {node['id']: node for node in nodes}

    return len(nodes_dict[node_id]['children']) == 0

