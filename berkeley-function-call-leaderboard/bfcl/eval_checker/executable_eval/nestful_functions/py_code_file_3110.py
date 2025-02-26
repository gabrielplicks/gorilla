from typing import Dict



def retrieve_city_from_data(data: Dict[str, Dict[str, str]]) -> str:

    """Retrieves the value associated with the key `city` in the nested dictionary `data`.



    Args:

        data: The nested dictionary containing the key `country_details`.



    Returns:

        The value associated with the key `city` in the nested dictionary, or the string `'Not Found'` if the key is not found.

    """

    try:

        return data["country_details"].get("city", "Not Found")

    except KeyError:

        return "Not Found"

