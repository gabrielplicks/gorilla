from typing import List

from xml.etree.ElementTree import Element



def get_genre(entries: List[Element]) -> List[str]:

    """Extracts the value of a specific XML attribute (`genre`) from a list of elements (`entries`).

    If the element does not contain the attribute, uses a default value of "None".



    Args:

        entries: A list of XML elements.



    Returns:

        A list of genre values.

    """

    return [entry.attrib.get('genre', 'None') for entry in entries]

