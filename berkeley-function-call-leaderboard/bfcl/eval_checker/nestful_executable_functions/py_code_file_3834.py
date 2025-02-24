from typing import List

from xml.etree import ElementTree



def merge_xml_elements(root_element_list: List[ElementTree.Element]) -> ElementTree.Element:

    """Merges a list of XML root elements into a single merged root element.



    Args:

        root_element_list: A list of XML root elements to merge.



    Returns:

        The merged root element.

    """

    merged_root = ElementTree.Element(root_element_list[0].tag)

    for root_element in root_element_list:

        for child in root_element:

            merge_xml_elements([child])

            merged_root.append(child)

    return merged_root

