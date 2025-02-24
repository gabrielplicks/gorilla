import xml.etree.ElementTree as ET

from typing import List, Dict



def extract_attributes(xml_string: str) -> List[Dict[str, str]]:

    """Extracts the attribute name and value from an XML string and returns a list of dictionaries.



    Args:

        xml_string: The XML string to extract attributes from.



    Returns:

        A list of dictionaries, each containing an attribute name and value.

        If the attribute value is missing, it is represented as an empty string.

    """

    root = ET.fromstring(xml_string)

    attributes = []



    def traverse(element):

        for attribute in element.attrib.items():

            attributes.append({"name": attribute[0], "value": attribute[1]})



        for child in element:

            traverse(child)



    traverse(root)



    return attributes

