from typing import List, Dict



def get_category_values(data: List[Dict], category: str) -> List:

    """Returns a list of the values of the `category` key in each dictionary.



    Args:

        data: A list of dictionaries.

        category: The key to extract values from the dictionaries.

    """

    return [d[category] for d in data]

