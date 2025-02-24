import os

from typing import Optional



def is_foo_bar_baz_set() -> bool:

    """Check whether the environment variable FOO_BAR_BAZ is set.



    Returns:

        bool: True if the environment variable is set, False otherwise.

    """

    return os.getenv('FOO_BAR_BAZ') is not None

