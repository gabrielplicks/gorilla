import xml.etree.ElementTree as etree



def xml_to_dict(xml_string: str) -> dict:

    """

    Generates a dictionary from an XML document. The keys of the dictionary are the

    XML elements' names, and the values are the corresponding attributes.

    Args:

        xml_string: The XML document as a string.

    """

    root = etree.fromstring(xml_string)

    return {element.tag: element.attrib for element in root}

