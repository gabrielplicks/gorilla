from typing import Dict, List



def bfs_shortest_path(graph: Dict[int, List[int]], start_node: int) -> Dict[int, int]:

    """Finds the shortest path from a starting node to all other nodes in a directed graph using BFS.



    Args:

        graph: The graph represented as an adjacency list.

        start_node: The starting node.



    Returns:

        A dictionary where each key is a node and the value is the distance to the node.

    """

    node_dist = {node: float('inf') for node in graph}

    node_dist[start_node] = 0

    queue = [start_node]



    while queue:

        node = queue.pop(0)

        for neighbor in graph[node]:

            if node_dist[neighbor] == float('inf'):

                node_dist[neighbor] = node_dist[node] + 1

                queue.append(neighbor)



    return node_dist

