import random

random.seed(42)
import string



def generate_random_string(length: int) -> str:

    """Generates a random string of lowercase alphanumeric characters and underscores.



    Args:

        length: The length of the random string to generate.



    Returns:

        A random string of lowercase alphanumeric characters and underscores.

    """

    all_characters = string.ascii_lowercase + string.digits + '_'

    def generate_random_character() -> str:

        random_character = random.choice(all_characters)

        return random_character.lower()  # Convert the character to lowercase

    random_string = ''.join(generate_random_character() for _ in range(length))



    return random_string

