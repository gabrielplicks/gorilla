from typing import List, Tuple



def count_connected_edges(graph: List[Tuple[int, int]], start_vertex: int) -> int:

    """Counts the number of connected edges in an undirected graph starting from the vertex with the given node ID.



    Args:

        graph: A list of edges representing the graph. Each edge is a tuple of two integers representing the vertices it connects.

        start_vertex: The ID of the vertex to start counting edges from.



    Returns:

        The number of edges connected to the start vertex.

    """

    return len([edge for edge in graph if edge[0] == start_vertex])

