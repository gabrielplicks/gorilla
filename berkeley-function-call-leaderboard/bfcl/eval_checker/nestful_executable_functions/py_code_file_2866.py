from typing import Dict, List



def assign_nodes_to_clusters(nodes: List[int], clusters: List[int]) -> Dict[int, int]:

    """Assigns each node from the given list to a cluster using a greedy approach.

    The function returns a mapping of cluster IDs to the assigned nodes.



    Args:

        nodes: A list of nodes to be assigned to clusters.

        clusters: A list of cluster IDs.



    Returns:

        A dictionary that maps each node to its assigned cluster.

    """

    node_clusters = {}

    cluster_sizes = {cluster_id: len(nodes) for cluster_id in clusters}



    for node in nodes:

        fewest_cluster_id = min(cluster_sizes, key=cluster_sizes.get)

        node_clusters[node] = fewest_cluster_id

        cluster_sizes[fewest_cluster_id] -= 1



    return node_clusters

