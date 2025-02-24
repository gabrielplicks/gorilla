from typing import Optional

from xml.etree import ElementTree



def find_by_id(xml_tree: ElementTree, search_id: str) -> Optional[ElementTree.Element]:

    """Finds the first XML element with the given ID in the given XML tree.



    Args:

        xml_tree: The XML tree to search.

        search_id: The ID to search for.



    Returns:

        The first XML element with the given ID, or None if no such element exists.

    """

    try:

        for element in xml_tree.iter():

            if element.get("id") == search_id:

                return element

        return None

    except Exception as e:

        print(f"Error: {e}")

        return None

