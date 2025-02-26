import json

from typing import List, Dict



def generate_json_from_enumerations(enumerations: List[Dict], keys: List[str]) -> str:

    """Generates a JSON serialized result from a list of dictionaries and a list of keys.

    The dictionaries represent a series of enumerations and the keys represent the name of the enumeration.

    The function returns a JSON serialized result with each key as an object name and each enumeration as its value.

    Args:

        enumerations: A list of dictionaries representing the enumerations.

        keys: A list of keys representing the name of the enumeration.

    """

    results = {}

    for dictionary in enumerations:

        for key in keys:

            if key in dictionary:

                if key in results:

                    results[key].append(dictionary[key])

                else:

                    results[key] = [dictionary[key]]

    return json.dumps(results)

