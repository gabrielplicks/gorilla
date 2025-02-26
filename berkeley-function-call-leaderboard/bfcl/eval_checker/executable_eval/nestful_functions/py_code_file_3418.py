import random

random.seed(42)
from typing import Union



def generate_code_sample(size: int, charset: str = '0123456789abcdefghijklmnopqrstuvwxyz', seed: Union[int, None] = None) -> str:

    """Generates a code sample of a given size by randomly selecting characters from a given character set using the provided seed (if any).



    Args:

        size: The size of the code sample to generate.

        charset: The character set to use for generating the code sample.

        seed: The seed to use for generating the random characters.

    """

    if seed is not None:

        random.seed(seed)



    assert size >= 0, "Size must be a non-negative integer"

    assert len(charset) > 0, "Charset must contain at least one character"



    return ''.join(random.choice(charset) for _ in range(size))

