from typing import List, Dict



def filter_gender(list_of_dicts: List[Dict], gender: str) -> List[Dict]:

    """Filters a list of dictionaries based on a given gender.

    Args:

        list_of_dicts: A list of dictionaries with keys `id`, `name`, `age`, and `gender`.

        gender: A string representing the gender to filter by.

    """

    return [d for d in list_of_dicts if d['gender'] == gender]

