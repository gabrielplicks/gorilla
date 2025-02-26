from typing import List, Dict

from xml.etree.ElementTree import Element



def get_attributes_list(root: Element) -> List[Dict[str, str]]:

    """

    Returns a list of list of attributes from each element in the XML tree.

    The first element in the list should be the list of attributes from the root element,

    while the remaining elements should be the attributes from the child elements, and so on.



    Args:

        root: The root element of the XML tree.

    """

    result = []

    def dfs(node):

        result.append(node.attrib)

        for child in node:

            dfs(child)

    dfs(root)

    return result

