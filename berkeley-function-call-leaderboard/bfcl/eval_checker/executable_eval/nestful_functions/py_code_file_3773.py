from typing import Dict, List



def sums_of_corresponding_numbers(data: Dict[str, List[int]]) -> Dict[str, List[int]]:

    """Calculates the sums of corresponding numbers in a dictionary of lists.



    Args:

        data: A dictionary of lists.



    Returns:

        A dictionary with a single key-value pair, where the key is the concatenated keys of the input dictionary and the value is a list of the sums of corresponding numbers.

    """

    return {

        '+'.join(data.keys()): [sum(num for num in nums) for nums in zip(*data.values())]

    }

