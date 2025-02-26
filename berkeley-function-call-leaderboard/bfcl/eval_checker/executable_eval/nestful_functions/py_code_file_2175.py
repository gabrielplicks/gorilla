from typing import Dict, List, Any



def create_map_by_category(objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:

    """Creates a dictionary where the keys are the categories and the values are lists of objects belonging to each category.



    Args:

        objects: A list of objects, where each object is a dictionary with a 'category' attribute.



    Returns:

        A dictionary where the keys are the categories and the values are lists of objects belonging to each category.

    """

    category_map = {}



    for obj in objects:

        if 'category' in obj:

            category = obj['category']

            if category in category_map:

                category_map[category].append(obj)

            else:

                category_map[category] = [obj]



    return category_map

