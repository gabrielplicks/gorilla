from typing import Dict



class Person:

    def __init__(self, name: str, age: int, gender: str):

        """Creates a person object with the given attributes.

        Args:

            name: The name of the person.

            age: The age of the person.

            gender: The gender of the person.

        """

        self.name = name

        self.age = age

        self.gender = gender



def create_person_object(attributes: Dict[str, str]) -> Person:

    """Creates a person object from a dictionary of attributes.

    Args:

        attributes: A dictionary containing the person's attributes.

    """

    person = Person(attributes['name'], attributes['age'], attributes['gender'])

    return person

