from typing import List, Tuple



def get_names_and_ages(lines: List[str]) -> List[Tuple[str, int]]:

    """Gets the names and ages from a list of lines in the format `Name, Age`.



    Args:

        lines: A list of lines in the format `Name, Age`.



    Returns:

        A list of tuples `(name, age)` where `name` is the string name in the line and `age` is the integer age in the line.

    """

    return [(name, int(age)) for line in lines for name, age in [line.split(', ')] if age.isdigit()]

