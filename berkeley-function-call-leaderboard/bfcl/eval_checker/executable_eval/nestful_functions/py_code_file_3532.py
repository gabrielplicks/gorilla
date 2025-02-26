import xml.etree.ElementTree as ET



def extract_id_from_xml_node(node: str) -> str:

    """Extracts the value of the `id` attribute from the first `b` element of the given XML node.



    Args:

        node: The XML node.



    Returns:

        The value of the `id` attribute of the first `b` element.

    """

    tree = ET.fromstring(node)

    for element in tree.iter("b"):

        return element.get("id")

