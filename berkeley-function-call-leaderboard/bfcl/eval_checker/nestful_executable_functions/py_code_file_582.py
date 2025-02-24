from typing import Dict, Optional



def get_food_info(food_dictionary: Dict, key: Optional[str] = 'name') -> tuple:

    """Extracts the name of a food item and its preparation time from a dictionary structure.



    Args:

        food_dictionary: The dictionary containing the food information.

        key: The key containing the food name. Defaults to 'name'.



    Returns:

        A tuple containing the food name and preparation time.

    """

    food_name = food_dictionary.get(key)

    preparation_time = food_dictionary.get('preparation_time')

    return food_name, preparation_time

