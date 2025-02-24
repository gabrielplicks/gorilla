from typing import Dict



def create_fake_object(properties: Dict[str, any]) -> object:

    """Creates a new object with the given attributes and values.

    Args:

        properties: A dictionary where the keys are the names of the attributes and the values are the corresponding values of the attributes.

    """

    obj_class = type("Object", (object,), properties)  # Dynamically create a new class with the given properties

    obj = obj_class()  # Create an instance of the new class

    for key, value in properties.items():

        setattr(obj, key, value)  # Set the attributes of the instance using setattr

    return obj

