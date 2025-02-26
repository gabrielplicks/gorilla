import json

import logging

from typing import List



def validate_json_fields(json_string: str, required_fields: List[str]) -> bool:

    """

    Validate the presence of specific fields in a JSON string.



    Args:

        json_string: A string containing a JSON object.

        required_fields: A list of fields to check for in the JSON object.



    Raises:

        ValueError: If any required field is missing in the JSON object.



    Returns:

        True if all required fields are present, False otherwise.

    """

    try:

        json_object = json.loads(json_string)

    except json.JSONDecodeError as err:

        logging.error(f"Error parsing JSON string: {err}")

        return False



    for field in required_fields:

        if field not in json_object:

            raise ValueError(f"Missing required field: {field}")



    return True

