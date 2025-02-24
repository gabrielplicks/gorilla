import importlib



def load_and_retrieve_attribute(module_name: str, attribute_name: str) -> object:

    """Dynamically loads a module by name and retrieves an attribute from the module.



    Args:

        module_name: The name of the module to load.

        attribute_name: The name of the attribute to retrieve from the module.



    Returns:

        The attribute from the module.

    """

    module = importlib.import_module(module_name)

    return getattr(module, attribute_name)

