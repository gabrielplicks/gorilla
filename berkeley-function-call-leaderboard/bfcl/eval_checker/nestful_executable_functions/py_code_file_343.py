from typing import Dict, List



def find_parents(graph: Dict[str, str], starting_node: str) -> List[str]:

    """Finds all the parents of a given node in a graph.



    Args:

        graph: A dictionary representing the graph, where keys are nodes and values are their parents.

        starting_node: The node to find the parents of.



    Returns:

        A list of all the parents of the given node.

    """

    parents = []

    current_node = starting_node



    while current_node in graph:

        parent = graph[current_node]

        parents.append(parent)

        current_node = parent

        if parent is None:

            break



    return parents

