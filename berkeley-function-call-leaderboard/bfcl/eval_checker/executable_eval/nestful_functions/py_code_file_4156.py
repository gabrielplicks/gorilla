from typing import List



def convert_string_to_integers(input_string: str) -> List[int]:

    """Converts an input string to a list of integers, where each integer represents the position of the corresponding letter in the alphabet.

    Args:

        input_string: The input string to be converted.

    Returns:

        A list of integers representing the alphabetical positions of the characters in the input string.

    """

    converted_integers = []

    for character in input_string:

        if character.isalpha():

            alphabetical_position = ord(character.lower()) - 96

            converted_integers.append(alphabetical_position)

        else:

            converted_integers.append(0)

    return converted_integers

