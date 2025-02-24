from typing import List, Dict, Any



def case_class(class_name: str, attr_names: List[str]) -> type:

    """Creates a class with the given name and attributes.



    Args:

        class_name: The name of the class to be created.

        attr_names: A list of attribute names.



    Returns:

        The created class object.

    """

    class_attrs: Dict[str, Any] = {attr: lambda self: getattr(self, attr) for attr in attr_names}

    return type(class_name, (), class_attrs)

