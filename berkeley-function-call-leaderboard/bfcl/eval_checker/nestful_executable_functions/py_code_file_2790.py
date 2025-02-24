import re

from typing import Dict



def parse_names_ages(names_ages: List[str]) -> Dict[str, int]:

    """Parses a list of strings representing names and ages into a dictionary.



    Args:

        names_ages: A list of strings, each representing a person's name and age in the format `name,age`.



    Returns:

        A dictionary where the keys are the names and the values are the ages. If the input is not in the correct format,

        the function returns an empty dictionary.

    """

    regex = re.compile(r"(\w+),(\d+)")

    name_age_dict = {}

    for name_age in names_ages:

        match = regex.match(name_age)

        if match:

            name, age = match.groups()

            name_age_dict[name] = int(age)

    return name_age_dict

