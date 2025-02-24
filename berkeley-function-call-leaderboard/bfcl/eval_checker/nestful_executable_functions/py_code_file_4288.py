from typing import List, Dict, Any

import collections



def create_edge_dict(edge_list: List[Tuple[Any, Any, Any]]) -> Dict[Any, List[Dict[str, Any]]]:

    """Creates a dictionary where each key is a node identifier and each value is a list of nodes connected to the key node.

    Each edge tuple is converted into a dictionary object with key-value pairs for the source and destination node identifiers,

    along with any other associated edge properties.



    Args:

        edge_list: A list of triplets representing edge connections.

    """

    edge_dict = collections.defaultdict(list)

    for src, dest, prop in edge_list:

        edge_dict[src].append({'src': src, 'dest': dest, 'prop': prop})

    return edge_dict

