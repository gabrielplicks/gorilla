import xml.etree.ElementTree as ET



def extract_text_between_fields(xml_string: str) -> str:

    """Extracts the text between the second and third `<field>` tags in an XML string.



    Args:

        xml_string: The XML string to parse.



    Raises:

        ValueError: If the second or third `<field>` tag is not found.

    """

    root = ET.fromstring(xml_string)

    second_field = root.find("./field[2]")

    if second_field is None:

        raise ValueError("No second field found")

    third_field = root.find("./field[3]")

    if third_field is None:

        raise ValueError("No third field found")

    return second_field.text + " " + third_field.text

