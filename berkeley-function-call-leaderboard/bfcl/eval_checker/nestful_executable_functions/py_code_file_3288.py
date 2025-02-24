from typing import Dict



def replace_lb_and_kg(dictionary: Dict[str, str]) -> None:

    """Replaces the values "lb" and "kg" in a dictionary with "pound" and "kilogram" respectively.



    Args:

        dictionary: A dictionary with string keys and values.

    """

    for index, key in enumerate(dictionary):

        value = dictionary.get(key)

        if value == "lb":

            dictionary[key] = "pound"

        elif value == "kg":

            dictionary[key] = "kilogram"

