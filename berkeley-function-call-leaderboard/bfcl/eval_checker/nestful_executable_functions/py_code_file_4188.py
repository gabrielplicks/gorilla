import xml.etree.ElementTree as ET



def calculate_sum_of_values(xml_string: str, tag_name: str) -> int:

    """Calculates the sum of the values of all attributes with the name 'value' for a given XML tag.



    Args:

        xml_string: A string containing XML content.

        tag_name: A string representing the name of an XML element.



    Returns:

        The sum of the values of all attributes with the name 'value' for the specified XML tag.

    """

    root = ET.fromstring(xml_string)

    elements = root.findall(tag_name)

    result = 0

    for element in elements:

        result += int(element.get('value'))



    return result

