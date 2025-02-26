from typing import Union



def get_fully_qualified_function_name(package_name: Union[str, None], function_name: str) -> str:

    """Creates a fully qualified function name from a package name and a function name.



    Args:

        package_name: The name of the package.

        function_name: The name of the function.



    Returns:

        A string that represents the fully qualified function name.

    """

    return f'{package_name}.{function_name}'

