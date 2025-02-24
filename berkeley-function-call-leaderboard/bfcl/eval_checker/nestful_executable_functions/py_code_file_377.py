from typing import List, Dict



def get_fields_from_schema(data: List[Dict[str, str]], schema: List[str]) -> List[Dict[str, str]]:

    """Filters a sequence of dictionaries based on a given schema.



    Args:

        data: A sequence of dictionaries.

        schema: A sequence of strings representing the keys to filter by.



    Returns:

        A sequence of filtered dictionaries.

    """

    return [{key: value for key, value in d.items() if key in schema} for d in data]

