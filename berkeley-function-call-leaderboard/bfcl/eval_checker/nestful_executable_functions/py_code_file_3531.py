from typing import Any, Tuple



def get_sorted_attributes(obj: Any) -> list[Tuple[str, Any]]:

    """Returns a list of tuples containing the names and values of the attributes of the given object, sorted in ascending order by attribute name.



    Args:

        obj: The object to extract attributes from.



    Returns:

        A list of tuples, where each tuple is a pair of strings containing the name of an attribute and its value.

    """

    attributes = []

    for attr in dir(obj):

        if not attr.startswith('_'):  # ignore dunder methods

            attributes.append((attr, getattr(obj, attr)))



    return sorted(attributes, key=lambda x: x[0])

