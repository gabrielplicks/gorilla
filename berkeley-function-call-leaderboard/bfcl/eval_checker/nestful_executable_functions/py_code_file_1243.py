from typing import Any



def get_object_representation(obj: Any) -> str:

    """Returns the string representation of the object in the format "Hi, my name is {{name}} and I live in {{city}}."

    Args:

        obj: The object with attributes "name" and "city".

    """

    name = obj.name

    city = "Unknown" if obj.city is None else obj.city

    return f"Hi, my name is {name} and I live in {city}."

