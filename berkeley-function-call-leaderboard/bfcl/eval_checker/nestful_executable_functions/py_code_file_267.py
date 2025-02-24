from typing import List



class User:

    def __init__(self, age: int):

        self.age = age



def get_average_age(users: List[User]) -> float:

    """Calculates the average age of a list of User objects.

    Args:

        users: A list of User objects.

    """

    total_age = 0

    for user in users:

        total_age += user.age

    average_age = total_age / len(users)

    return average_age

