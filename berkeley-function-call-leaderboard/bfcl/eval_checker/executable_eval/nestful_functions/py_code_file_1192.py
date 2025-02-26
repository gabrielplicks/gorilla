from typing import List, Dict, Tuple



def create_new_list_of_tuples(people: List[Dict[str, str]]) -> List[Tuple[str, int]]:

    """Creates a new list of tuples containing only the name and age of each person.



    Args:

        people: A list of dictionaries where each dictionary represents a person with their name and age.



    Returns:

        A new list of tuples containing only the name and age of each person.

    """

    return [(person['name'], person['age']) for person in people]

