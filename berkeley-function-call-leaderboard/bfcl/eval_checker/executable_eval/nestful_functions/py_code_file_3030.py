from typing import Dict



def dict_of_kwargs(**kwargs: Dict[str, bool]) -> Dict[str, str]:

    """Converts arbitrary keyword arguments into a dictionary.

    If the keyword argument is a boolean type, it is converted to a string following the format "keyword=True" or "keyword=False".

    Otherwise, it is directly converted to a string.

    Args:

        kwargs: Arbitrary keyword arguments.

    """

    result = {}



    for key, value in kwargs.items():

        if isinstance(value, bool):

            result[key] = f'{key}={str(value).lower()}'

        else:

            result[key] = str(value)



    return result

