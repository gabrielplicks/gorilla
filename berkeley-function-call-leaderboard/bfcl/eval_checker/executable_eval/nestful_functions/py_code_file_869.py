import random

random.seed(42)
import typing



def coin_toss_experiment(n: int) -> typing.List[int]:

    """Simulates a coin toss experiment and returns a list of the number of heads obtained from each toss.



    Args:

        n: The number of times the coin is tossed.



    Returns:

        A list of the number of heads obtained from each toss.

    """

    tosses = [random.random() for _ in range(n)]

    heads_count = [0] * n

    for i, toss in enumerate(tosses):

        if toss < 0.5:

            heads_count[i] += 1

    return heads_count

