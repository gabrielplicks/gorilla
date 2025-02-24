import json

from typing import List



def parse_json_file(file_path: str) -> List[str]:

    """Parses a JSON file containing a list of dictionaries and extracts the `email` and `updated_at` key-value pairs.



    Args:

        file_path: The path to the JSON file.



    Returns:

        A list of strings in the format `email:updated_at`.

    """

    with open(file_path, "r") as f:

        data = json.load(f)

    results = []

    for item in data:

        email = item.get("email", "")

        updated_at = item.get("updated_at", "")

        results.append(f"{email}:{updated_at}")

    return results

