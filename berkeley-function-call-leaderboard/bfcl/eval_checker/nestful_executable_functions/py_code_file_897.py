from typing import Any, Tuple



def check_class_attributes(cls: Any, attr_list: list[Tuple[str, Any]]) -> bool:

    """Checks if the class attributes match the key-value pairs in the list.



    Args:

        cls: The class object.

        attr_list: A list of two-element tuples containing attribute names and expected values.



    Returns:

        True if all the attributes match and False if any attribute does not match.

    """

    for attr, value in attr_list:

        if getattr(cls, attr) == value:

            continue

        else:

            return False

    return True

