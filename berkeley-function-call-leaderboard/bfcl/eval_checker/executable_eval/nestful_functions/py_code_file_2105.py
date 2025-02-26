import xml.etree.ElementTree as ET

from typing import List



def get_all_elements_by_tag_name(xml_node: ET.Element, tag_name: str) -> List[ET.Element]:

    """

    Returns a list of all the child elements with a specific tag name.

    If the XML node has no child elements with the given tag name, the function returns an empty list.

    The function also handles the case where the input XML node is None or empty.



    Args:

        xml_node: The XML node to search for child elements.

        tag_name: The tag name of the child elements to search for.

    """

    if xml_node is None or len(xml_node) == 0:

        return []

    return xml_node.findall(f".//{tag_name}")

