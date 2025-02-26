from typing import Any



def get_class_and_id(obj: Any) -> tuple:

    """Extracts the class name and instance id of any given object in the Python runtime.



    Args:

        obj: The object to extract the class name and instance id from.



    Returns:

        A tuple containing the class name and instance id.

    """

    class_name = type(obj).__name__

    instance_id = id(obj)

    return (class_name, instance_id)

