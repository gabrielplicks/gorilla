from typing import List, Any



class Person:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



peeps = [

    Person("Alice", 25),

    Person("Bob", 30),

    Person("Charlie", 28)

]



def get_instance_attribute_values(instance_list: List[Any], attribute: str) -> List[Any]:

    """Returns a list of all the values of a specific data attribute for each instance.



    Args:

        instance_list: A list of class instances.

        attribute: The name of the data attribute to return the values for.

    """

    return [getattr(instance, attribute) for instance in instance_list]



names = get_instance_attribute_values(peeps, 'name')

print(names) # Output: ["Alice", "Bob", "Charlie"]

