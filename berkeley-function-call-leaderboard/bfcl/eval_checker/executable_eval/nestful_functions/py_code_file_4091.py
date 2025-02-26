from typing import Dict

from xml.etree import ElementTree



def extract_xml_values(xml_doc: str) -> Dict[str, str]:

    """Extracts the values of the XML elements in a given XML document.



    Args:

        xml_doc: The XML document as a string.



    Returns:

        A dictionary with the element name as the key and the value as the corresponding value.

    """

    root = ElementTree.fromstring(xml_doc)

    values = {}

    for element in root.iter():

        values[element.tag] = element.text

    return values

