from typing import Union



def greet_world_or_name(name: Union[str, None]) -> str:

    """Creates a function that takes in a string as an argument and returns either "Hello, World!" or "Hello, {name}!".



    Args:

        name: The input string.



    Returns:

        A greeting string.

    """

    if name == "World":

        return "Hello, World!"

    else:

        return f"Hello, {name}!"

