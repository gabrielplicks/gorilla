import xml.etree.ElementTree as ET



def xml_to_string(xml_tree: ET.Element) -> str:

    """Converts an XML tree to a string representation.



    Args:

        xml_tree: The XML tree to convert.



    Raises:

        ValueError: If the XML tree is invalid.

    """

    try:

        xml_string = ET.tostring(xml_tree, encoding='unicode')

    except ET.ParseError:

        raise ValueError('Invalid XML tree')



    return xml_string

