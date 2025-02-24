from typing import Any



def is_all_true(*args: Any) -> bool:

    """Checks if all arguments are true.



    Args:

        args: Any number of arguments to check.

    """

    for arg in args:

        if not arg:

            return False

    return True

