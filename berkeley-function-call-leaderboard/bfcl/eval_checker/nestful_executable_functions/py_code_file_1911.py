import random

random.seed(42)
from typing import List



def generate_signal(length: int) -> str:

    """Generates a binary signal of given length.

    The generated signal consists of randomly generated 1-bit signals,

    where each signal has a 10% chance of being a 1 (true) and a 90% chance

    of being a 0 (false). The signal is represented as a string of 1s and 0s,

    with each character representing a signal.



    Args:

        length: The length of the signal to generate.



    Returns:

        A string of 1s and 0s representing the generated signal.

    """

    signal = ''

    for _ in range(length):

        if random.random() < 0.1:

            signal += '1'

        else:

            signal += '0'

    return signal

