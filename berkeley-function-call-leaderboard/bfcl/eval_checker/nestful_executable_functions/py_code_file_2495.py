from typing import List



def round_to_bin_edge(value: float, bin_edges: List[float]) -> float:

    """Rounds a value to the nearest bin edge.



    Args:

        value: The value to round.

        bin_edges: A list of bin edges.

    """

    bin_edges = [float('-inf')] + bin_edges + [float('inf')]

    for idx in range(len(bin_edges) - 1):

        if bin_edges[idx] <= value < bin_edges[idx + 1]:

            return bin_edges[idx]

