from typing import Dict



def get_phone_number(name: str, phonebook: Dict[str, str]) -> str:

    """

    Returns the phone number of the person if the person is in the phonebook,

    or a string indicating that the person is not found in the phonebook.



    Args:

        name: The name of the person.

        phonebook: A dictionary that maps names to phone numbers.

    """

    if name in phonebook:

        return phonebook[name]

    else:

        return f'The person "{name}" is not found in the phonebook.'

