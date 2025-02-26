from typing import List, Dict, Any



def get_attribute_values_for_methods(objects: List[Any]) -> Dict[str, List[Any]]:

    """Creates a dictionary where the keys are the names of the objects' methods, and the values are the corresponding list of arguments for those methods.



    Args:

        objects: A list of objects.



    Returns:

        A dictionary where the keys are the names of the objects' methods, and the values are the corresponding list of arguments for those methods.

    """

    method_names = ['method_a', 'method_b']

    attribute_values = {}



    for method_name in method_names:

        attribute_values[method_name] = [getattr(obj, method_name)() for obj in objects]



    return attribute_values

