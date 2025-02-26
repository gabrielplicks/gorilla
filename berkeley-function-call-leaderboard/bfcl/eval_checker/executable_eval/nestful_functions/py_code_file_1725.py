from typing import Any



def is_same_source(obj1: Any, obj2: Any) -> bool:

    """Checks if two objects have the same `source` attribute and are different from `None`.



    Args:

        obj1: The first object.

        obj2: The second object.



    Returns:

        True if both objects have the same `source` attribute and are different from `None`. Otherwise, False.

    """

    return obj1.source == obj2.source and obj1 is not None and obj2 is not None

