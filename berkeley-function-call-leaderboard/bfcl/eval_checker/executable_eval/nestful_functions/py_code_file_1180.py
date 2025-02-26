import json

from typing import List, Dict, Any



def convert_json_to_dict_list(json_str: str) -> List[Dict[str, Any]]:

    """

    Converts a JSON-formatted string to a list of dictionaries.

    Args:

        json_str: The JSON-formatted string to convert.

    Returns:

        A list of dictionaries, where each dictionary corresponds to a row in the input JSON data.

    """

    data = json.loads(json_str)



    if isinstance(data, dict):

        return [data]



    output = []

    for obj in data:

        if isinstance(obj, dict):

            output.append(obj)

        else:

            output.append({k: v for k, v in obj.items()})



    return output

