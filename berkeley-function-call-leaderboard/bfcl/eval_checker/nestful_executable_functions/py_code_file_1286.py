from typing import Any, Dict, List, Tuple



def get_attributes_from_object(obj: Any) -> List[Tuple[str, Any]]:

    """Returns a list of tuples containing the attribute names and their values from the given object.



    Args:

        obj: The object to extract attributes from.



    Returns:

        A list of tuples, where each tuple contains the attribute name (as a string) and its value.

    """

    attributes: Dict[str, Any] = vars(obj)

    return [(key, attributes[key]) for key in attributes]

