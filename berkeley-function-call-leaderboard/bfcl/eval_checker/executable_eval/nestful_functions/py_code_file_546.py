from typing import Any



def check_one_attribute(obj: Any, attr: str) -> bool:

    """Checks whether an object contains an attribute.



    Args:

        obj: The object to check.

        attr: The attribute to check for.



    Returns:

        True if the attribute is found, False otherwise.

    """

    if not isinstance(obj, object):

        print("obj is not an object")

        return False

    if not isinstance(attr, str):

        print("attr is not a string")

        return False



    if hasattr(obj, attr):

        print(f"Object {obj} has attribute {attr}")

        return True

    else:

        print(f"Object {obj} does not have attribute {attr}")

        return False

