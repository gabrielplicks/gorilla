from typing import Dict, Any



def make_class(class_name: str, attr_dict: Dict[str, Any]) -> type:

    """Creates a new class object with the given class name and attributes.



    Args:

        class_name: The name of the new class.

        attr_dict: A dictionary of attributes for the new class.



    Returns:

        A new class object with the given class name and attributes.

    """

    return type(class_name, (object,), attr_dict)

