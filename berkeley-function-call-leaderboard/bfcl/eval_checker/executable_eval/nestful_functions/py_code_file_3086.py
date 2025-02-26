from typing import List, Dict



def average_age_of_active_people(people: List[Dict[str, str]]) -> float:

    """Calculates the average age of all people whose status is "Active".



    Args:

        people: A list of dictionaries representing different JSON objects. Each dictionary has `id`, `name`, `age`, `status` keys.



    Returns:

        The average age of all people whose status is "Active".

    """

    active_people = [person for person in people if person["status"] == "Active"]

    return sum(person["age"] for person in active_people) / len(active_people)

