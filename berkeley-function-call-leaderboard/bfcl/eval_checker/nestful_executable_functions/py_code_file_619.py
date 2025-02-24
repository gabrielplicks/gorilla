from typing import Union



def info(name: str, age: Union[int, float]) -> str:

    """Returns a formatted string containing the name and age.



    Args:

        name: The name of the person.

        age: The age of the person.

    """

    return f"Name: {name}, Age: {age}"

