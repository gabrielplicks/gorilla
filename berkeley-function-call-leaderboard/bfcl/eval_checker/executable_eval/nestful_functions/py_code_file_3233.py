from typing import List, Any



def add_node_to_graph(g: List[List[Any]], n: Any) -> int:

    """Adds a node to a graph and returns the index of the newly added node.

    Args:

        g: The graph data structure represented by a list of lists.

        n: The node to be added to the graph.

    """

    if not g:

        g.append([n])

        return 0

    g.append([n])

    return len(g) - 1

