import xml.etree.ElementTree as ET



def get_elements_text(xml_file: str, query: str) -> list[str]:

    """Extracts the text of all the elements from an XML document that match a given XPath query.



    Args:

        xml_file: The name of the XML file.

        query: The XPath query to select the elements from the document.



    Returns:

        A list of strings representing the text of all the selected elements.

    """

    tree = ET.parse(xml_file)

    elements = tree.findall(query)

    return [elem.text for elem in elements]

