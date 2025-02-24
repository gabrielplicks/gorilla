import random

random.seed(42)
import string



def generate_passwords(num_characters: int, num_passwords: int) -> list:

    """Generates a list of random passwords.



    Args:

        num_characters: The number of characters in each password.

        num_passwords: The total number of passwords to generate.



    Returns:

        A list of random passwords.

    """

    passwords = []

    for _ in range(num_passwords):

        password = ''.join(random.choices(string.ascii_letters + string.digits, k=num_characters))

        passwords.append(password)

    return passwords

