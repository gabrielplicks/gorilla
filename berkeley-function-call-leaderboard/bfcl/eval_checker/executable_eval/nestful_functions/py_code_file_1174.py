from typing import Optional



def extract_pub_type_from_xml(xml_str: str) -> Optional[str]:

    """Extracts the publication type from an XML string.



    Args:

        xml_str: The XML string to extract the publication type from.



    Returns:

        The publication type if found, otherwise None.

    """

    if "article" in xml_str:

        return "article"

    elif "book" in xml_str:

        return "book"

    elif "patent" in xml_str:

        return "patent"

    else:

        return None

