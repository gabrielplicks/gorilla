from typing import Dict



def remove_all_edges(graph: Dict[str, list]) -> Dict[str, list]:

    """Removes all edges of a graph represented as a dictionary.

    Args:

        graph: The graph dictionary, where each key is a node and each value is a list of the nodes connected to it.

    """

    new_graph = {}

    for key, value in graph.items():

        new_graph[key] = []

    return new_graph

