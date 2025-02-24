import xml.etree.ElementTree as ET

from typing import Dict, List



def parse_xml_enum(xml_file: str) -> Dict[str, List[Dict[str, str]]]:

    """Parses an XML file and extracts the `type`, `name`, and `value` attributes of all `enum` elements.

    Returns a dictionary with the element's `type` as the key and a list of dictionaries containing `name` and `value` attributes as the value.

    Args:

        xml_file: The path to the XML file.

    """

    root = ET.parse(xml_file).getroot()

    enum_dict = {}

    for enum_elem in root.findall('enum'):

        enum_type = enum_elem.get('type')

        enum_dict[enum_type] = [elem.attrib for elem in enum_elem.findall('e')]

    return enum_dict

