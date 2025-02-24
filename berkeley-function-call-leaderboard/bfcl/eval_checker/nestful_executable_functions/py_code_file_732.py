import json

from unittest import mock

from typing import Dict



def download_and_deserialize(url: str) -> Dict:

    """Downloads a JSON file from a specified URL and deserializes it.

    If an error occurs, an empty dictionary is returned.

    Args:

        url: The URL of the JSON file to download.

    """

    try:

        response = mock.get(url)

        response.raise_for_status()

        data = json.loads(response.content)

        return data

    except Exception:

        return {}

