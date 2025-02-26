from typing import Dict



class Person:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def create_person_json(name: str, age: int) -> Dict[str, str]:

    """Creates a Person object with the given name and age and returns a JSON object containing the name and age properties.



    Args:

        name: The name of the person.

        age: The age of the person.



    Returns:

        A JSON object containing the name and age properties.

    """

    person = Person(name, age)

    return {"name": person.name, "age": person.age}

