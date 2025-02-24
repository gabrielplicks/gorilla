from typing import Any



class Person:

    pass



def is_valid_person(obj: Any) -> bool:

    """Determines if a given object is a valid instance of a `Person` class and has a `name` attribute.



    Args:

        obj: The object to check.



    Returns:

        True if the object is a valid instance of a `Person` class and has a `name` attribute, False otherwise.

    """

    return isinstance(obj, Person) and hasattr(obj, 'name')

