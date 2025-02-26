from typing import Union



def display_name(first_name: str, last_name: str) -> None:

    """Displays the full name of a person given their first and last names.



    Args:

        first_name: The first name of the person.

        last_name: The last name of the person.



    Raises:

        ValueError: If either the first name or last name is an empty string.

    """

    if first_name == "" or last_name == "":

        raise ValueError("First name and last name must not be empty strings.")



    full_name = first_name + " " + last_name

    print(full_name)

