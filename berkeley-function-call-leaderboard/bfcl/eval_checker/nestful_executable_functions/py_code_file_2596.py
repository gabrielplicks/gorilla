import json



def get_water_level(json_string: str) -> float:

    """

    Extracts the value of the key "water_level" from the "data" field in a JSON string.



    Args:

        json_string: A JSON string containing the data.

    """

    water_data = json.loads(json_string)

    return water_data["data"]["water_level"]

