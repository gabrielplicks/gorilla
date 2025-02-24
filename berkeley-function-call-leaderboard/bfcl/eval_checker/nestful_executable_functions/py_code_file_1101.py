from typing import Any



def handle_scenario(obj: Any, string: str) -> bool:

    """Checks if a string exists in an object's list of strings.



    Args:

        obj: The object containing the list of strings.

        string: The string to check for in the object's list of strings.



    Raises:

        Exception: If the object is missing the field.

    """

    if not hasattr(obj, "field"):

        raise Exception("Object is missing field.")



    for item in obj.field:

        if item == string:

            return True



    return False

