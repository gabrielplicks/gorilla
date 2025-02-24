import json

from typing import Dict



def organize_json(json_string: str) -> Dict[str, object]:

    """Organizes a JSON string into a more readable and easier to work with object.



    Args:

        json_string: The JSON string to be organized.



    Returns:

        A dictionary containing the type and value of the JSON data.

    """

    data = json.loads(json_string)

    organized_data = {

        'type': type(data).__name__,

        'value': data

    }



    return organized_data

