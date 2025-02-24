import random

random.seed(42)
import typing



def get_attribute_name(dictionary: typing.Dict[str, typing.Any]) -> str:

    """Selects a random attribute name from a dictionary object.



    Args:

        dictionary: The dictionary object to select an attribute name from.



    Returns:

        A randomly selected attribute name from the dictionary object.

    """

    return random.choice(list(dictionary.keys()))

