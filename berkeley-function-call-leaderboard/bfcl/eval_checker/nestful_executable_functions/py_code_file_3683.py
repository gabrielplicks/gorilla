from typing import Dict



def translate_dictionary(dictionary: Dict[str, str]) -> Dict[str, str]:

    """Translates a dictionary where the keys are stored as a string of "key0, key1, key2, ..., keyN",

    and the values are stored as a string of "value0, value1, value2, ..., valueN".



    Args:

        dictionary: The dictionary to be translated.



    Returns:

        The translated dictionary.

    """

    keys = dictionary["keys"].split(", ")

    values = dictionary["values"].split(", ")

    translated_dictionary = dict(zip(values, keys))



    return translated_dictionary

