from typing import Dict, List



def filter_by_schema(data: List[Dict], schema: Dict) -> List[Dict]:

    """Filters a list of dictionaries based on a schema.



    Args:

        data: A list of dictionaries to filter.

        schema: A dictionary representing the schema to filter by.



    Returns:

        A filtered list of dictionaries that match the schema.

    """

    def is_valid(dict: Dict) -> bool:

        return all(k in dict and dict[k] == v for k, v in schema.items())



    return [d for d in data if is_valid(d)]

