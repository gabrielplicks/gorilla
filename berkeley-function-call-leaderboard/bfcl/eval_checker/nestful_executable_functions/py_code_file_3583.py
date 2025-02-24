import random

random.seed(42)


def die_roll() -> int:

    """

    Simulates a die roll and returns the resulting number.

    Args:

        None

    """

    number = random.randint(1, 6)



    if number <= 4:

        return 1

    else:

        return 2

