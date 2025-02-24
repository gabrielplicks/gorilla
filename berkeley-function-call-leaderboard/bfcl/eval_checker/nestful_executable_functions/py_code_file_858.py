from typing import List, Tuple



def map_colors(colors: List[Tuple[int, int, int]], mapping_func: callable) -> List[Tuple[int, int, int]]:

    """Maps a given set of RGB colors to a new set of RGB colors.



    Args:

        colors: A list of RGB tuples representing a set of colors.

        mapping_func: A function that takes an RGB tuple and returns a new tuple.



    Returns:

        A new list of RGB tuples representing the transformed colors.

    """

    return [mapping_func(c) for c in colors]

