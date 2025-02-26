import json

import requests



def request_api_data(url: str) -> dict:

    """

    Makes an API request to the provided URL and returns the response as a Python

    dictionary. If there is any error, an empty dictionary is returned.

    Args:

        url: The URL to make the API request to.

    """

    try:

        response = requests.get(url)  # Make the API request

        response.raise_for_status()  # Raise an exception if the request fails

        data = response.json()  # Parse the response body as JSON and convert to Python dictionary

        return data

    except Exception as e:

        print(f"Error in request_api_data: {e}")

        return {}

