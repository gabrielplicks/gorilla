import copy



def get_recursively_defined_objects(definitions: dict) -> dict:

    """Returns a dictionary containing all recursively defined objects.

    This function copies the input dictionary deeply and does not modify the original input dictionary.

    Args:

        definitions: A dictionary of definitions.

    """

    recursively_defined_objects: dict = {}



    for key, value in definitions.items():

        if type(value) == dict:

            recursively_defined_objects[key] = copy.deepcopy(value)



    return recursively_defined_objects

