from typing import List



def underscored_list(input_list: List[int]) -> str:

    """

    Returns a string with the integers in the input list separated by underscores.

    If the input list is empty, it returns an empty string.

    If the input list contains non-integer elements, it returns the string "None".

    If the input list contains negative numbers, it returns the string "Negative".

    If an exception occurs during the execution of the function, it returns the string "Error".



    Args:

        input_list: A list of integers.

    """

    try:

        if not input_list:

            return ""

        elif any(not isinstance(element, int) for element in input_list):

            return "None"

        elif any(element < 0 for element in input_list):

            return "Negative"

        else:

            return "_".join(str(element) for element in input_list)

    except Exception:

        return "Error"

