import random

random.seed(42)
import typing



def simulate_coin_toss(n: int) -> typing.List[int]:

    """Simulates `n` coin tosses and returns a list of 0s and 1s, where 0 represents tails and 1 represents heads.



    Args:

        n: The number of coin tosses to simulate.



    Returns:

        A list of 0s and 1s representing the results of the coin tosses.

    """

    return [random.randint(0, 1) for _ in range(n)]

