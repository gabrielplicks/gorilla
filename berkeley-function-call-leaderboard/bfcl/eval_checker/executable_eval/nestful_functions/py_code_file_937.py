import random

random.seed(42)
from typing import List, Dict



def find_longest_name(people: List[Dict[str, str]]) -> str:

    """Finds and returns the name of the person with the longest name in a list of people's information.



    Args:

        people: A list of dictionaries, where each dictionary contains a person's name and age.



    Returns:

        The name of the person with the longest name, formatted as "Name: ".



    Raises:

        ValueError: If the list is empty.

    """

    if not people:

        raise ValueError("Error: Empty list")



    longest_name_length = max(len(person["name"]) for person in people)

    longest_names = [person["name"] for person in people if len(person["name"]) == longest_name_length]

    return random.choice(longest_names)

