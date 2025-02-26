from typing import List



def predict_next_value(sequence: List[int], n_steps: int) -> int:

    """Predicts the next value in the given sequence using the most recent two values.

    The function is modular and calculates a single step of the forecast.

    Args:

        sequence: The sequence of numbers.

        n_steps: The number of steps to forecast.

    """

    window = sequence[-2:]

    for _ in range(n_steps):

        next_value = window[-1] + window[-2]

        window.append(next_value)

    return window[-1]

