from typing import Union



def remove_namespaces(name: str) -> Union[str, None]:

    """Removes the namespace prefix from an XML element name.

    Args:

        name: The XML element name.

    """

    parts = name.split(":")

    if len(parts) > 1:

        return parts[1]

    else:

        return name

