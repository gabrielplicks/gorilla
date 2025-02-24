import random

random.seed(42)


def generate_name() -> str:

    """Generates a random name using the given rules.



    Args:

        None



    Returns:

        A string representing the generated name.

    """

    first_letter = chr(random.randint(65, 90))  # A-Z

    second_letter = chr(random.randint(65, 90))  # A-Z

    third_letter = str(random.randint(0, 9))  # 0-9

    fourth_letter = chr(random.randint(97, 122))  # a-z

    return first_letter + second_letter + third_letter + fourth_letter

