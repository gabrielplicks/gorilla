from typing import Any



class Location:

    def __init__(self, location: str):

        self.location = location



def get_address(obj: Any) -> str:

    """Gets the address from an object.

    Args:

        obj: The object to get the address from.

    Returns:

        The address attribute of the object if it exists, the location attribute if the object is an instance of a class that inherits from the Location class, or None if neither of these conditions are met.

    """

    if hasattr(obj, 'address'):

        return obj.address

    elif isinstance(obj, Location):

        return obj.location

    else:

        return None

