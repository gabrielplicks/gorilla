import xml.etree.ElementTree as ET

from typing import List, Optional



def extract_item_value(xml_doc: str) -> Optional[List[str]]:

    """Extracts the "value" attribute of the "item" tag from an XML document.



    Args:

        xml_doc: The XML document as a string.



    Returns:

        A list of "value" attribute values if successful, otherwise None.

    """

    try:

        root = ET.fromstring(xml_doc)

    except ET.ParseError as e:

        print("Invalid XML document:", e)

        return None



    item_values = []



    for item in root.findall('item'):

        if 'value' not in item.attrib or not item.attrib['value']:

            print(f"Missing or empty value attribute in item: {item}")

            return None



        item_values.append(item.attrib['value'])



    return item_values

