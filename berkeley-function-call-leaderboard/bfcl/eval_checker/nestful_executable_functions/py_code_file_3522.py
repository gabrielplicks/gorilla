from typing import List



def get_partition_names(name_pattern: str, numbers: List[int]) -> List[str]:

    """Constructs a list of strings by concatenating `name_pattern` with each number from `numbers`, separated by a hyphen.



    Args:

        name_pattern: The string to be concatenated with each number.

        numbers: The list of numbers to be concatenated with `name_pattern`.



    Returns:

        A list of strings, each of which is constructed by concatenating `name_pattern` with a number from `numbers`, separated by a hyphen.

    """

    partition_names = []



    for number in numbers:

        partition_name = f"{name_pattern}-{number}"

        partition_names.append(partition_name)



    return partition_names

