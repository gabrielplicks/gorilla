from typing import List, Union



def pytype_list(obj: Union[List, type, object]) -> List[str]:

    """

    Returns a list of type names for the input object.

    Args:

        obj: The input object.

    """

    if isinstance(obj, list):

        return [pytype_list(item) for item in obj]

    if isinstance(obj, type):

        return [obj.__name__]

    return [type(obj).__name__]

