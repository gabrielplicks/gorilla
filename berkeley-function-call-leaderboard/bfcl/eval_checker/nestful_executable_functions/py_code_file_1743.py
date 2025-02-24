from typing import List, Dict



def find_paths(graph: Dict[str, List[str]], source: str, destination: str) -> List[List[str]]:

    """Finds all possible paths from a source node to a destination node in a directed graph.



    Args:

        graph: A dictionary representing the directed graph. Each key is a node and its value is a list of nodes that can be reached directly from the key node.

        source: The source node.

        destination: The destination node.



    Returns:

        A list of paths, each path is a list of nodes representing the path from the source to the destination.

    """

    paths = []

    def dfs(current: str, path: List[str]):

        if current == destination:

            paths.append(path)

        else:

            for neighbor in graph[current]:

                dfs(neighbor, path + [neighbor])



    dfs(source, [source])

    return paths

