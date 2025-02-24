from typing import Optional



class Person:

    def __init__(self, favourite_food: str, friends: list):

        self.favourite_food = favourite_food

        self.friends = friends



def find_person_with_favourite_food(person: Person) -> Optional[Person]:

    """Finds the first person who has the same favourite food as the input person.



    Args:

        person: The input person instance.



    Returns:

        The first person who has the same favourite food as the input person, or None if no such person exists.

    """

    if not hasattr(person, "favourite_food"):

        return None



    for other_person in person.friends:

        if hasattr(other_person, "favourite_food") and other_person.favourite_food == person.favourite_food:

            return other_person



    return None

