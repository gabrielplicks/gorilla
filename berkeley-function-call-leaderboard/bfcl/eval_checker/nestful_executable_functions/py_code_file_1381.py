from typing import Dict, Any



def extract_values_from_class_attribute(objects: Dict[str, Any], class_attribute: str, default_value: Any) -> list:

    """Extracts a list of values from a specific class attribute in a dictionary of objects.

    If the attribute is missing, returns the default value instead.



    Args:

        objects: A dictionary of objects.

        class_attribute: The class attribute to extract values from.

        default_value: The default value to return if the attribute is missing.

    """

    return [

        getattr(obj, class_attribute, default_value)

        for obj in objects.values()

    ]

