import random

random.seed(42)


def generate_magic_number() -> int:

    """Generates a randomly generated positive integer between 1 and 100 without using any library functions.

    The function also prints to the console using string interpolation and an appropriate message indicating the generated number.

    Returns:

        The generated integer.

    """

    magic_number = random.randint(1, 100)

    print(f"Generated magic number: {magic_number}")

    return magic_number

