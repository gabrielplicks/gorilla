from typing import Dict, Union



def convert_value(value: Union[Dict, str]) -> Union[Dict, str]:

    """Converts a value to a JSON-encodable value.



    Args:

        value: The value to convert.



    Returns:

        The converted value.

    """

    if isinstance(value, dict):

        return to_json(value)

    elif isinstance(value, str):

        return "str:" + value

    else:

        return value



def to_json(d: Dict[str, Union[Dict, str]]) -> Dict[str, Union[Dict, str]]:

    """Converts a nested dictionary to a JSON-compatible dictionary.



    Args:

        d: The nested dictionary to convert.



    Returns:

        The converted dictionary.

    """

    return {k: convert_value(v) for k, v in d.items()}

