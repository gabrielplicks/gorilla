from typing import List



class Person:

    def __init__(self, name: str, age: int, gender: str, country: str):

        self.name = name

        self.age = age

        self.gender = gender

        self.country = country



def get_person_names(people: List[Person]) -> List[str]:

    """Returns a list of names of people that are older than 20 years old.



    Args:

        people: A list of Person class instances.



    Returns:

        A list of strings representing the names of people with an age greater than 20.

    """

    return [person.name for person in people if person.age > 20]

