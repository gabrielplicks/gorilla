from typing import List



class Person:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def generate_formatted_string(people: List[Person]):

    """Generates a formatted string from a list of Person instances.

    Args:

        people: A list of Person instances.

    """

    formatted_strings = []

    for person in people:

        formatted_strings.append(f"{person.name}, age {person.age}")

    return "\n".join(formatted_strings)



people = [

    Person("John", 27),

    Person("Mary", 25),

    Person("Lucy", 20),

]

print(generate_formatted_string(people))

