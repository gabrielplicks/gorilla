from typing import Union



def format_pip_command(package_name: Union[str, int]) -> str:

    """Formats a pip install command with the given package name and version.



    Args:

        package_name: The name of the package to install.

    """

    return f"pip install {package_name}=={package_name}"

