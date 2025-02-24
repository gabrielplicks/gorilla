from typing import Any

import json

import datetime



def convert_recorded_at_to_timestamp(json_file: str) -> str:

    """Converts the `recorded_at` field from a string to a timestamp in a JSON file.



    Args:

        json_file: The path to the JSON file.



    Returns:

        A string that represents the converted JSON object.

    """

    with open(json_file, "r") as f:

        json_data: dict[str, Any] = json.load(f)



    date_string: str = json_data["recorded_at"]

    date: datetime.datetime = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")

    timestamp: int = int(date.timestamp())

    json_data["recorded_at"] = timestamp



    return json.dumps(json_data)

