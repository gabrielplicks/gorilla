from typing import Dict



def build_person(first_name: str, last_name: str, age: int) -> Dict[str, str | int]:

    """Builds a person dictionary with the given first_name, last_name, and age.



    Args:

        first_name: The first name of the person.

        last_name: The last name of the person.

        age: The age of the person.



    Returns:

        A dictionary with keys 'first_name', 'last_name', and 'age'.

    """

    person = {

        'first_name': first_name,

        'last_name': last_name,

        'age': age

    }

    return person

