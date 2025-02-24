import string

import random

random.seed(42)


def generate_captcha() -> str:

    """Generate a random captcha consisting of 4 letters, each of which is randomly chosen from the English alphabet.



    Returns:

        A 4-character string.

    """

    captcha = ""

    for _ in range(4):

        captcha += random.choice(string.ascii_letters)

    return captcha

