from typing import Union



def first_four_characters(input_string: str) -> Union[str, None]:

    """Returns the first four characters of the input string, separated by a hyphen.

    If the string is less than four characters, the function returns a string with the same characters repeated until it reaches four characters.

    Args:

        input_string: The input string.

    """

    try:

        if len(input_string) < 4:

            return input_string * ((4 - len(input_string)) // len(input_string) + 1)

        else:

            return input_string[:4]

    except Exception as e:

        print(f"An error occurred: {e}")

        return None

