from typing import List



def validate_list(lst: List[int], output_file: str) -> bool:

    """Validates whether a list of integers satisfies a certain property.



    Args:

        lst: A list of integers.

        output_file: The path to an output file.



    Returns:

        A boolean value indicating whether the list satisfies the property.

    """

    if len(lst) < 3:

        with open(output_file, 'w') as f:

            f.write("Failed!")

        return False

    if not all(x >= 0 for x in lst):

        with open(output_file, 'w') as f:

            f.write("Failed!")

        return False

    with open(output_file, 'w') as f:

        f.write("Passed!")

    return True

