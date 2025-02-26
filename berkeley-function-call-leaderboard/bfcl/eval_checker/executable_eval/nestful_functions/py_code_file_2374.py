from typing import Any, Dict, List



def count_container_items(container: Any) -> int:

    """Counts the number of items in a container, such as a list or a dictionary.

    If the container is an empty one, returns 0. If it is not a container, returns None instead.

    Args:

        container: The container to count the items of.

    """

    if isinstance(container, List):

        return len(container)

    elif isinstance(container, Dict):

        return len(container)

    else:

        return None

