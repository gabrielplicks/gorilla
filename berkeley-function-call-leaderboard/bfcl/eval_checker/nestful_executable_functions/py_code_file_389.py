from typing import AnyStr



def normalize_input_and_get_greeting_message(input_string: AnyStr) -> AnyStr:

    """Normalizes the input string and returns a greeting message.



    Args:

        input_string: The input string to be normalized.



    Returns:

        A greeting message based on the normalized input string.

    """

    while True:

        normalized_input = input_string.lower().strip().replace("  ", " ")

        if normalized_input == "hello":

            return "Hello, World!"

        elif normalized_input == "goodbye":

            return "Goodbye, World!"

        else:

            input_string = input("Enter a new string: ")

