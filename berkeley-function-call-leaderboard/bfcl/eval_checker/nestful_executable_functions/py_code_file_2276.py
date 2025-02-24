from typing import Optional



def short_hostname(node_name: str) -> Optional[str]:

    """Returns the short hostname from a given node name.

    The short hostname is the substring before the first hyphen (-) or the entire string if no hyphen is present.

    Args:

        node_name: The node name string.

    """

    index = node_name.find('-')

    if index == -1:

        return node_name

    return node_name[:index]

