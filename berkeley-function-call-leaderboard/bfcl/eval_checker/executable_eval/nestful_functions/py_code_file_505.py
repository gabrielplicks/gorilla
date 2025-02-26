from typing import Any



def check_and_set(data: Any, attr1: str, attr2: str) -> None:

    """Checks if the object has the `attr1` attribute. If so, creates a new attribute `attr2` and sets its value to the value of `attr1`, otherwise does nothing.



    Args:

        data: The JSON object.

        attr1: The attribute to check for.

        attr2: The attribute to set.

    """

    if hasattr(data, attr1):

        setattr(data, attr2, getattr(data, attr1))

