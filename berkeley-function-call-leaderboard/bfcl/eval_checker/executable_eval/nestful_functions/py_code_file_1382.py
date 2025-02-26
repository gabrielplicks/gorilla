import importlib



def load_class_from_path(path: str) -> type:

    """Loads a class from a string path.



    Args:

        path: The full path to the module, including the package name and the file name.



    Returns:

        The class object.

    """

    module_name, class_name = path.rsplit('.', 1)

    module = importlib.import_module(module_name)

    class_obj = getattr(module, class_name)



    return class_obj

