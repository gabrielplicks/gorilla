import json

from typing import List, Dict



def read_persons(file_path: str) -> List[Dict[str, str]]:

    """Reads a JSON file and returns a list of dictionaries representing persons.



    Args:

        file_path: The path to the JSON file.



    Returns:

        A list of dictionaries representing persons. Each dictionary has the keys "name", "age", "height", and "weight".

    """

    with open(file_path) as file:

        data = json.load(file)



    persons = []

    for person in data:

        person_dict = {

            "name": person["name"],

            "age": str(person["age"]),

            "height": str(person["height"]),

            "weight": str(person["weight"]),

        }

        persons.append(person_dict)



    return persons

