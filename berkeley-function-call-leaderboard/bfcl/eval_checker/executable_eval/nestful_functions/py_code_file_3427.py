from typing import List, Optional



class Person:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def get_youngest_person(people: List[Person]) -> Optional[str]:

    """Returns the name of the youngest person in a list of `Person` objects.



    Args:

        people: A list of `Person` objects.



    Returns:

        The name of the youngest person, or an empty string if the list is empty.

    """

    if not people:

        return ''



    youngest_age = float('inf')  # Initialize with a large value

    youngest_person = None



    for person in people:

        if person.age < youngest_age:

            youngest_age = person.age

            youngest_person = person.name



    return youngest_person

