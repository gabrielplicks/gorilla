from typing import List, Dict



def count_rolls(rolls: List[Dict[str, int]], numbers: List[int]) -> List[int]:

    """

    Counts the occurrences of each number in the given list of numbers that occurred in the rolls.



    Args:

        rolls: A list of dictionaries where each dictionary represents a dice roll.

        numbers: A list of integers between 1 and 6.



    Returns:

        A list of the counts of each number in numbers that occurred in the rolls.

    """

    return [sum(1 for roll in rolls if roll['die1'] == number or roll['die2'] == number) for number in numbers]

