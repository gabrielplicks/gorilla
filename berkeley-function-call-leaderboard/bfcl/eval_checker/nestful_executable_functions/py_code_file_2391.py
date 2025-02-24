from typing import Dict, Any



def recursive_compose(d: Dict[Any, Any]) -> Dict[Any, Any]:

    """Composes a dictionary with itself recursively.



    Args:

        d: The input dictionary.



    Returns:

        A dictionary of dictionaries.



    Raises:

        ValueError: If the input is not a dictionary or if a key is not a string, integer, or dictionary.

    """

    if not isinstance(d, dict):

        raise ValueError('Input must be a dictionary')



    composed_dict = {}

    for k, v in d.items():

        if isinstance(k, str) or isinstance(k, int):

            composed_dict[k] = v

        elif isinstance(k, dict):

            composed_dict[k] = recursive_compose(k)

        else:

            raise ValueError(f'Unsupported key type: {type(k).__name__}')



    return composed_dict

