from typing import List



def convert_to_class_names(input_string: str) -> List[str]:

    """Converts a string to a list of class names.



    Each word in the input string is treated as a separate class name. The first letter of each word is capitalized, and the suffix 'Class' is added.



    Args:

        input_string: The input string to convert to class names.



    Returns:

        A list of class names.

    """

    words = input_string.split()

    class_names = [word.capitalize() + 'Class' for word in words]

    return class_names

