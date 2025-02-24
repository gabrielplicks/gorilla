import xml.etree.ElementTree as ET

from typing import List



def extract_articles(xml_string: str) -> List[ET.Element]:

    """Traverses the Document Object Model (DOM) of an XML document and extracts all tags with the name "article".

    Args:

        xml_string: The XML document as a string.

    Returns:

        A list of XML tags.

    """

    root = ET.fromstring(xml_string)

    return root.findall("article")

