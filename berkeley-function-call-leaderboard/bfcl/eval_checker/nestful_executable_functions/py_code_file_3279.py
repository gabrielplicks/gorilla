from typing import Dict, List, Any



def find_sln_attributes(obj: Dict[str, Any], attributes: List[str] = []) -> List[str]:

    """Recursively finds all attributes of an object that start with "sln" (case-insensitive).



    Args:

        obj: The object to search for attributes.

        attributes: The list of attributes found so far.



    Returns:

        The list of attributes that start with "sln".

    """

    for key, value in obj.items():

        if key.lower().startswith("sln"):

            attributes.append(key)

        if isinstance(value, dict):

            find_sln_attributes(value, attributes)

    return attributes

