import xml.etree.ElementTree as ET

from typing import List



def extract_keyword_ids(xml_doc: str, keyword_list: List[str]) -> List[str]:

    """Extracts the IDs of specific keywords from an XML document.



    Args:

        xml_doc: The XML document as a string.

        keyword_list: A list of keywords to search for in the XML document.



    Returns:

        A list of IDs that match the keywords in the XML document.

    """

    root = ET.fromstring(xml_doc)

    ids = []

    for item in root.iter('item'):

        keywords = item.attrib['keywords'].split(', ')

        if any(keyword in keywords for keyword in keyword_list):

            ids.append(item.attrib['id'])

    return ids

