from typing import Dict, List



def get_nodes_with_attribute(graph: Dict[str, Dict[str, str]], attribute_name: str, attribute_value: str) -> List[str]:

    """Returns a list of all nodes in the graph that have an attribute with the specified name and value.



    Args:

        graph: A dictionary of dictionaries representing a graph where each node has a dictionary of attributes.

        attribute_name: The name of the attribute to search for.

        attribute_value: The value of the attribute to match.

    """

    matching_nodes = []

    for node, attributes in graph.items():

        if attribute_name in attributes and attributes[attribute_name] == attribute_value:

            matching_nodes.append(node)

    return matching_nodes

