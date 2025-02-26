from typing import List



class Person:

    pass



def filter_persons_by_country(persons: List[Person], country_code: str) -> List[Person]:

    """Filters a list of Person objects by country code.



    Args:

        persons: A list of Person objects.

        country_code: The country code to filter by.



    Returns:

        A list of Person objects whose address field's country code is the provided country code.

    """

    return [person for person in persons if person.address.country_code == country_code]

