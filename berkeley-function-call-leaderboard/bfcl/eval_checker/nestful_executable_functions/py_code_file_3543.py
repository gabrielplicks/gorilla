from typing import List



def validate_element(num: float) -> bool:

    """Validates a number by checking if it is greater than zero.



    Args:

        num: The number to validate.



    Returns:

        True if the number is valid, False otherwise.

    """

    if num > 0:

        return True

    else:

        return False



def validate_list(nums: List[float]) -> bool:

    """Validates a list of numbers by mapping the `validate_element` function to the list and checking that all elements are valid.



    Args:

        nums: The list of numbers to validate.



    Returns:

        True if all numbers in the list are valid, False otherwise.

    """

    validation_result = list(map(validate_element, nums))

    if False in validation_result:

        return False

    else:

        return True

