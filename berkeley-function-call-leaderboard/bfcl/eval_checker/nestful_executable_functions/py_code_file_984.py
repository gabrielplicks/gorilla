from typing import Dict, List, Union



def map_salaries(people: List[Dict[str, Union[str, int]]]) -> Dict[str, int]:

    """Creates a new dictionary that maps each person's name to their salary.

    If the person's salary is unknown, use a value of 0.

    Args:

        people: A list of dictionaries, each representing a person.

    """

    return {person['name']: person.get('salary', 0) for person in people}

