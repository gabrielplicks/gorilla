from typing import Any, List



def get_attributes_by_namespace(obj: Any, namespace: str) -> List[str]:

    """Gets all attributes from a Python object that match a specific namespace.



    Args:

        obj: The object to get attributes from.

        namespace: The namespace string to match attributes against.

    """

    attributes = []

    for attribute in dir(obj):

        if attribute.startswith(namespace):

            attributes.append(attribute)

    return attributes

