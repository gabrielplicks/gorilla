from typing import Dict, Any



def create_object_with_attributes(class_: type, attributes: Dict[str, Any]):

    """Creates an object of the input class with the specified attributes.

    Args:

        class_: The class to create an object of.

        attributes: A dictionary of attributes to assign to the object.

    """

    instance = class_()

    for key, value in attributes.items():

        setattr(instance, key, value)

    return instance

