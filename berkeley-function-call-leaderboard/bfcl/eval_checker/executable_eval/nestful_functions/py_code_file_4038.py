from typing import List, Any



def get_attr_from_obj_list(obj_list: List[Any], attr_name: str) -> List[Any]:

    """Retrieves the values of a specified attribute from a list of objects.



    Args:

        obj_list: A list of objects.

        attr_name: A string representing the name of the attribute to retrieve.



    Returns:

        A list of the attribute values for each object in `obj_list`. If an object

        does not have the specified attribute, `None` is returned for that object.

    """

    attr_values = []

    for obj in obj_list:

        try:

            attr_value = getattr(obj, attr_name)

        except AttributeError:

            attr_value = None

        attr_values.append(attr_value)

    return attr_values

