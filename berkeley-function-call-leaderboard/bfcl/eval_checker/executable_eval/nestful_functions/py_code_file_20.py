from typing import Dict, Any



def add_feature(dictionary: Dict[str, list[str]], filename: str) -> Dict[str, list[str]]:

    """Adds a feature to a dictionary from a file if the key matches.



    Args:

        dictionary: The dictionary to add the feature to.

        filename: The name of the file to load the feature from.

    """

    with open(filename) as file:

        for line in file:

            key, value = line.split()



            if key in dictionary:

                dictionary[key].append(value)



    return dictionary

