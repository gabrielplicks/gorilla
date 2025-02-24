import json

from typing import List



def get_unique_languages(json_file_path: str) -> List[str]:

    """Extracts a list of all unique languages from a JSON file containing a list of package objects.

    Each package has a `language` attribute.

    Args:

        json_file_path: The path to the JSON file.

    Returns:

        A list of unique languages.

    """

    try:

        with open(json_file_path) as json_file:

            data = json.load(json_file)

            unique_languages = set()

            for package in data['packages']:

                unique_languages.add(package['language'])

            return list(unique_languages)

    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:

        print(f"Error reading the JSON file: {e}")

        return []

