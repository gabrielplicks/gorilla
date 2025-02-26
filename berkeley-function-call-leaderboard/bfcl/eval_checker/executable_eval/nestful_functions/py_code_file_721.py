import xml.etree.ElementTree as ET

from typing import Dict



def count_tag_names(xml_element: ET.Element, tag_name: str) -> Dict[str, int]:

    """

    Counts the occurrences of a given tag name in an XML document.



    Args:

        xml_element: The root element of the XML document.

        tag_name: The tag name to count.



    Returns:

        A dictionary containing the tag name and its corresponding count.

    """

    tag_counts = {}



    def traverse_elements(element: ET.Element):

        if element.tag == tag_name:

            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1



        for child_element in element:

            traverse_elements(child_element)



    traverse_elements(xml_element)

    return tag_counts

