from typing import Dict



def validate_and_return_person_info(first_name: str, last_name: str, age: int) -> Dict[str, str]:

    """Validates the input and returns a dictionary containing the given information.



    Args:

        first_name: The first name of the person.

        last_name: The last name of the person.

        age: The age of the person.



    Raises:

        ValueError: If the input is invalid.

    """

    if not isinstance(first_name, str):

        raise ValueError('first_name must be a string')

    if not isinstance(last_name, str):

        raise ValueError('last_name must be a string')

    if not isinstance(age, int) or age < 0:

        raise ValueError('age must be a positive integer')

    return {'firstName': first_name, 'lastName': last_name, 'age': age}

