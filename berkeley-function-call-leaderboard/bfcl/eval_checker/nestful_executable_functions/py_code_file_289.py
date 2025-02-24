import xml.etree.ElementTree as ET

from typing import List, Dict



def convert_xml_to_json(xml_path: str) -> List[Dict[str, str]]:

    """Converts an XML file to a JSON object.



    Args:

        xml_path: The path to the input XML file.



    Returns:

        A list of dictionaries, where each dictionary contains the field values of an `element` tag.

        If the input XML file is empty, the function returns an empty list.

    """

    tree = ET.parse(xml_path)

    root = tree.getroot()

    if not root:

        return []

    json_objects = []

    for element in root:

        element_dict = {}

        for field in element:

            element_dict[field.tag] = field.text

        json_objects.append(element_dict)

    return json_objects

