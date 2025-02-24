from typing import Any, Dict



def json_to_python_dict(json_obj: Dict[str, Any]) -> Dict[str, Any]:

    """Converts a JSON object (dictionary) to a Python dictionary.



    Args:

        json_obj: The JSON object (dictionary) to be converted.



    Returns:

        A Python dictionary corresponding to the input JSON object.

    """

    python_dict = {}

    for key, value in json_obj.items():

        if isinstance(value, dict):

            python_dict[key] = json_to_python_dict(value)

        elif isinstance(value, list):

            python_dict[key] = [json_to_python_dict(item) for item in value]

        else:

            python_dict[key] = value

    return python_dict

