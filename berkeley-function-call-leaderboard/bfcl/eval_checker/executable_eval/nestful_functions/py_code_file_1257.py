from typing import List

import requests



def get_asset_names(client: requests.Session, endpoint_url: str) -> List[str]:

    """Retrieves the asset names from the "assets" endpoint of a RESTful API service.



    Args:

        client: An HTTP client instance.

        endpoint_url: The API endpoint URL.



    Returns:

        A list of asset names.

    """

    response = client.get(endpoint_url)

    response_json = response.json()

    assets = response_json['assets']

    asset_names = [asset['name'] for asset in assets]

    return asset_names

