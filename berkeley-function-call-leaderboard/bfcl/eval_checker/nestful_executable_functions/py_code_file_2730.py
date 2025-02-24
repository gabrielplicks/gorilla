from typing import List



def generate_class_names(module_name: str) -> List[str]:

    """Generates a list of class names from a specified module's public attributes using `__import__()` and `getattr()`.



    Args:

        module_name: The name of the module to generate class names from.



    Returns:

        A list of class names.

    """

    module = __import__(module_name)

    public_attributes = [attr for attr in dir(module) if not attr.startswith('_')]

    class_names = [attr for attr in public_attributes if isinstance(getattr(module, attr), type)]



    return class_names

