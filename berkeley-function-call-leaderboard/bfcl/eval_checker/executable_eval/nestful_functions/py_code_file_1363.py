from typing import List



def generate_name_list(names: List[str], count: int) -> List[str]:

    """Generates a list of strings, each string representing a list of names, where the names are concatenated together with their corresponding index number.



    Args:

        names: A list of strings representing names of people.

        count: An integer representing the number of people to generate a list of.



    Returns:

        A list of strings, each string representing a list of names, where the names are concatenated together with their corresponding index number.

    """

    if count < 1 or not names:

        return []

    name_list = [str(i+1) + '. ' + name for i, name in zip(range(count), names)]

    if len(name_list) < count:

        name_list += name_list * (count // len(name_list))



    return name_list[:count]

