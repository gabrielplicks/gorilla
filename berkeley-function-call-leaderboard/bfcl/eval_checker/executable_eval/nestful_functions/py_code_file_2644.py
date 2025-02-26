from typing import List, Dict



def instantiate_objects(class_instances: List[Dict[str, str]]) -> List[object]:

    """Instantiates objects from a list of dictionaries representing class names and attributes.



    Args:

        class_instances: A list of dictionaries, where each dictionary contains a

            "class_name" key with the name of the class to be instantiated, and an

            "attributes" key with a dictionary of keyword arguments to be passed to

            the class constructor.



    Returns:

        A list of objects created from the class names and attributes.

    """

    return [

        class_data["class_name"](**class_data["attributes"])

        for class_data in class_instances

    ]

