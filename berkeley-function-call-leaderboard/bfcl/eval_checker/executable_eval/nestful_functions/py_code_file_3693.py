from typing import List, Dict



def sum_prices(json_objects: List[Dict[str, float]]) -> float:

    """Calculates the sum of the 'price' field in a list of JSON objects.



    Args:

        json_objects: A list of JSON objects.



    Returns:

        The sum of the 'price' field in the JSON objects.

    """

    if not json_objects:

        return 0



    total = 0

    for obj in json_objects:

        if obj.get('price') is not None:

            total += obj['price']

    return total

