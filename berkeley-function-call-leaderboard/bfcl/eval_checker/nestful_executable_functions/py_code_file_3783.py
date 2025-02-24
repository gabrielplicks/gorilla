from typing import List

from xml.etree import ElementTree as ET



def get_xlink_href(element_tree: ET.ElementTree) -> List[str]:

    """Returns a list of unique href attribute values for all XLink elements in the document.



    Args:

        element_tree: The ElementTree object representing the document.



    Returns:

        A list of unique href attribute values.

    """

    href_values = []



    for element in element_tree.iter():

        if element.tag.endswith('}XLink'):

            href_value = element.get('href', '')  # If 'href' is not present, the default is an empty string

            if href_value not in href_values:

                href_values.append(href_value)



    return href_values

