import importlib



def load_package(package_name: str) -> object:

    """Loads a package given its name.



    Args:

        package_name: The name of the package to be loaded.



    Returns:

        The package object.

    """

    package = importlib.import_module(package_name)

    return package

