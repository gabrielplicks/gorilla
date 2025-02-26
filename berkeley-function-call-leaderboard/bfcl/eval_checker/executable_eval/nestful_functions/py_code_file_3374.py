import inspect

from typing import Dict, List, Tuple



def get_class_meta(cls: type) -> Dict[str, List[str]]:

    """Returns a dictionary containing all metadata for a given class.



    Args:

        cls: The class object.



    Returns:

        A dictionary containing the name of the class and the names of all methods.

    """

    methods: List[Tuple[str, type]] = inspect.getmembers(cls, inspect.isfunction)

    meta: Dict[str, List[str]] = {

        'name': cls.__name__,

        'methods': [method[0] for method in methods]

    }

    return meta

