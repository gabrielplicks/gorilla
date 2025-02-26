from typing import Any



last_result: Any = None  # Define the global variable



def store_last_result(string: str) -> None:

    """Stores the given string as a global variable named `last_result`.



    Args:

        string: The string to be stored.

    """

    global last_result

    last_result = string

