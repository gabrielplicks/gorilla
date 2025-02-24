from typing import Any, Dict



def get_attributes_as_dict(instance: Any) -> Dict[str, Any]:

    """Returns a dictionary containing the names and values of all public attributes of a class instance.



    Args:

        instance: A class instance.



    Returns:

        A dictionary with the attribute names and values.

    """

    return {name: value for name, value in vars(instance).items()}

