from typing import Dict



def remove_duplicate_entries_from_json(json_object: Dict[str, list]) -> Dict[str, int]:

    """Removes duplicate entries from a JSON object and returns the updated JSON object as a Python dict.

    If no duplicate entries are found, it returns the original JSON object without changes.

    Args:

        json_object: The JSON object to remove duplicate entries from.

    """

    assert isinstance(json_object, dict), "Expected a JSON object, but got a different type."



    for key, values in json_object.items():

        json_object[key] = min(values)



    return json_object

