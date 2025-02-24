from typing import Optional



import sys



def get_runtime_environment() -> str:

    """Returns the name of the current Python runtime environment.



    Args:

        None



    Returns:

        str: The name of the current Python runtime environment.

    """

    if sys.ps1 is not None:

        if get_ipython() is not None:

            return "IPython"

        else:

            return "Python Console"

    else:

        return "Non-Interactive"

