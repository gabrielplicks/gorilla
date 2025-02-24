from typing import Dict

import xml.dom.minidom



def parse_xml_to_dict(xml_string: str) -> Dict[str, str]:

    """Parses an XML string and returns a dictionary representing the relevant data.



    Args:

        xml_string: The XML string to be parsed.



    Returns:

        A dictionary containing the root element's attributes and child elements' tag and text content.

    """

    ret_dict: Dict[str, str] = {}

    xml_doc = xml.dom.minidom.parseString(xml_string)

    root_element = xml_doc.documentElement

    for attribute in root_element.attributes.items():

        ret_dict[attribute[0]] = attribute[1]

    for element in root_element.childNodes:

        if element.nodeType == element.ELEMENT_NODE:

            ret_dict[element.tagName] = element.firstChild.nodeValue



    return ret_dict

