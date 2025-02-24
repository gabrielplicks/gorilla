from typing import Dict, Any



class Person:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def get_class_properties(cls: Any) -> Dict[str, Any]:

    """Returns a dictionary of class property names and values.

    Args:

        cls: The class instance.

    Returns:

        A dictionary of property names and values.

    """

    properties = vars(cls)

    return properties



person = Person("John", 30)

print(get_class_properties(person))

