import json

from typing import List, Dict



def extract_urls_from_input(input_structure: str) -> List[str]:

    """Extracts a list of URLs from a JSON-like structure.



    Args:

        input_structure: A JSON-like structure containing a list of dictionaries with `id` and `url` keys.



    Returns:

        A list of URLs that satisfy the following conditions:

        - The `id` value must be greater than 100

        - The `url` value must contain the substring `https://`

    """

    data: List[Dict[str, str]] = json.loads(input_structure)

    filtered_urls: List[str] = [

        dict_data["url"]

        for dict_data in data

        if dict_data["id"] > 100 and "https://" in dict_data["url"]

    ]



    return filtered_urls

