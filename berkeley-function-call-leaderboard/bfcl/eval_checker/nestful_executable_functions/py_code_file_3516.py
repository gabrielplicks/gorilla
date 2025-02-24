from typing import Dict, List



def get_connected_nodes(graph: Dict[str, List[str]], start_node: str) -> List[str]:

    """Returns all nodes connected to the given node in the graph.



    Args:

        graph: A dictionary representing the graph. Each key is a node, and the value is a list of nodes connected to that node.

        start_node: The node to start the search from.



    Returns:

        A list of nodes connected to the start_node.

    """

    stack = [start_node]

    visited = set()

    connected_nodes = []



    while stack:

        node = stack.pop()



        if node not in visited:

            visited.add(node)

            connected_nodes.append(node)



            for neighbor in graph[node]:

                if neighbor not in visited:

                    stack.append(neighbor)



    return connected_nodes

