import itertools

from typing import List, Tuple



def sort_edges_by_type(edges: List[Tuple[int, int, str]]) -> List[List[Tuple[int, int, str]]]:

    """Sorts a list of edges according to their type.



    The edges are represented as tuples of (src_node, dst_node, type), where type is a string.

    The function returns a list of lists, where each sublist contains edges of a specific type.



    Args:

        edges: A list of edges represented as tuples (src_node, dst_node, type).



    Returns:

        A list of lists, where each sublist contains edges of a specific type.

    """

    sorted_edges = sorted(edges, key=lambda edge: edge[2])

    grouped_edges = itertools.groupby(sorted_edges, key=lambda edge: edge[2])

    result = [list(group) for _, group in grouped_edges]



    return result

