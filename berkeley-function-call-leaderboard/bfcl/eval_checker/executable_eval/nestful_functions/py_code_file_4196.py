import random

random.seed(42)


def gen_random_string(length: int) -> str:

    """Generates a random string of a fixed length, using the ASCII printable character set.



    Args:

        length: The length of the string to be generated.



    Returns:

        A string object of the random string.

    """

    ascii_printable_start = 33

    ascii_printable_end = 126

    random_string = ""

    for _ in range(length):

        random_char_code = random.randint(ascii_printable_start, ascii_printable_end)

        random_char = chr(random_char_code)

        random_string += random_char

    return random_string

