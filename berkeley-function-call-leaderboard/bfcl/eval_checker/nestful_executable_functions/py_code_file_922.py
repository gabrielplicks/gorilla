from typing import Dict, List



def get_unique_class_names(classes: Dict[str, object]) -> List[str]:

    """

    Retrieves a list of unique class names from a dictionary of class names and instances.

    If a class is not found, the function returns an empty list.

    Args:

        classes: A dictionary containing class names as keys and instances as values.

    """

    try:

        class_names = list(classes.keys())

        return list(set(class_names))

    except Exception as e:

        print(f"Error: {e}")

        return []

