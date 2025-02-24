from typing import Any



def is_hashable(obj: Any) -> bool:

    """Checks if the given object is hashable.



    Args:

        obj: The object to check.



    Returns:

        True if the object is hashable, False otherwise.

    """

    try:

        hash(obj)

        return True

    except AttributeError:

        return False

    except Exception:

        return False

