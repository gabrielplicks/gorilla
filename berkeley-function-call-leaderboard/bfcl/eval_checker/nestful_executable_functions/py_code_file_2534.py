import random

random.seed(42)
import typing



def generate_six_digit_id() -> str:

    """Generates a random 6-digit identification (ID) number in the format of "{number}-{number}-{number}", where each number is a random integer in the range of 0 to 99.



    Args:

        None



    Returns:

        A string representing the generated 6-digit ID number.

    """

    number1: int = random.randint(0, 99)

    number2: int = random.randint(0, 99)

    number3: int = random.randint(0, 99)

    id_string: str = f"{number1}-{number2}-{number3}"

    return id_string

