from typing import Any, Dict, List



def collect_attrs(obj: Any, attrs: List[str]) -> Dict[str, Any]:

    """Collects attributes from a given object in a dictionary.



    Args:

        obj: The object to collect attributes from.

        attrs: A list of attribute names.



    Returns:

        A dictionary where each attribute name is the key and the corresponding value is the value of that attribute in the object. If an attribute is not present in the object, the value for that attribute is `None`.

    """

    result = {}

    for attr in attrs:

        result[attr] = getattr(obj, attr, None)

    return result

