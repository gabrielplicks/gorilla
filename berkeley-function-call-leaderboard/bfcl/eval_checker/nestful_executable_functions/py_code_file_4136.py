from typing import List



def generate_neighbor_list(graph: List[List[int]]) -> List[List[int]]:

    """Generates a neighbor list from a directed graph represented as an adjacency list.



    Args:

        graph: A directed graph represented as an adjacency list.



    Returns:

        A list of lists representing the neighbor list of the graph.

    """

    neighbor_list = [[] for _ in range(len(graph))]

    for node, neighbors in enumerate(graph):

        for neighbor in neighbors:

            neighbor_list[node].append(neighbor)



    return neighbor_list

