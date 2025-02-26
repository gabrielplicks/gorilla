from typing import List



def add_values_to_attribute(obj: object, attr_name: str, values: List) -> None:

    """Adds values to an attribute of an object.



    If the attribute exists already, the function adds all the values to the existing attribute.

    If the attribute does not exist, the function creates the attribute and assigns the list of values to it.



    Args:

        obj: The object to which the attribute belongs.

        attr_name: The name of the attribute.

        values: The values to be added to the attribute.

    """

    if hasattr(obj, attr_name):

        current_value = getattr(obj, attr_name)

        updated_value = current_value + values

    else:

        updated_value = values

    setattr(obj, attr_name, updated_value)

