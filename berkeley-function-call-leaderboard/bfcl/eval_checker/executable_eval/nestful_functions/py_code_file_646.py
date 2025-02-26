import xml.etree.ElementTree as ET



def extract_tags_and_values(xml_string: str) -> dict:

    """Parses an XML string and extracts all the tags and their corresponding values.



    Args:

        xml_string: The XML string to parse.



    Returns:

        A dictionary containing each tag-value pair, where the key is the tag name and the value is its value.

    """

    root = ET.fromstring(xml_string)

    tags_and_values = {}

    for element in root:

        tags_and_values[element.tag] = element.text

    return tags_and_values

