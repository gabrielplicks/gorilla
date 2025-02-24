import random

random.seed(42)
from typing import List



def generate_chromosome(length: int) -> List[int]:

    """Generates a random chromosome, represented as a list of integers of a given length.

    Each integer is a random value between 0 and 20 that is unique within the chromosome.

    Args:

        length: The length of the chromosome.

    """

    numbers = set()

    chromosome = []

    while len(numbers) < length:

        num = random.randint(0, 20)

        if num not in numbers:

            numbers.add(num)

            chromosome.append(num)



    random.shuffle(chromosome)

    return chromosome

