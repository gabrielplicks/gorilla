from typing import List, Dict



def filter_by_country(people: List[Dict], country: str, include: bool = True) -> List[Dict]:

    """Filters a list of dictionaries representing people based on a specified country and a boolean value.



    Args:

        people: A list of dictionaries representing people.

        country: The name of the country to filter by.

        include: A boolean value indicating whether to include or exclude people from the specified country.

    """

    filtered = []

    for person in people:

        if person["country"] == country and include:

            filtered.append(person)

        elif person["country"] != country and not include:

            filtered.append(person)

    return filtered

