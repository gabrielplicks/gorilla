from typing import Tuple



def minutes_difference(t1: str, t2: str) -> str:

    """Calculates the difference in minutes between two time strings in the format "HH:MM".

    The function returns a string in the format "HH:MM".

    Args:

        t1: The first time string in the format "HH:MM".

        t2: The second time string in the format "HH:MM".

    """

    h1, m1 = t1.split(':')

    h2, m2 = t2.split(':')

    minutes_1 = int(h1) * 60 + int(m1)

    minutes_2 = int(h2) * 60 + int(m2)

    minutes_diff = abs(minutes_1 - minutes_2)

    hours_diff = minutes_diff // 60

    minutes_diff %= 60

    return f"{hours_diff:02d}:{minutes_diff:02d}"

