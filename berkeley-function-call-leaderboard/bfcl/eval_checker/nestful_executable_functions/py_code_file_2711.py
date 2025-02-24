from typing import List



class User:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def filter_users_by_age(users: List[User]) -> List[str]:

    """Filters a list of users by their age and returns a list of strings containing their name and age category.



    Args:

        users: A list of User objects.



    Returns:

        A list of strings containing the user's name and age category.

    """

    filtered_users = []



    for user in users:

        if user.age > 18:

            age_category = "Adult"

        elif 13 <= user.age <= 17:

            age_category = "Teenager"

        else:

            age_category = "Child"



        filtered_users.append(f"{age_category}: {user.name}")



    return filtered_users

