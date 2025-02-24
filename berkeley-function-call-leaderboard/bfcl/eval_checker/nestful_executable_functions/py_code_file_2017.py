from typing import Any



def reload_page(page: Any) -> bool:

    """Reloads a page object if it has a .reload() method.



    Args:

        page: The page object to reload.



    Returns:

        True if the page was reloaded, False otherwise.



    Raises:

        TypeError: If the input is not an object.

    """

    if isinstance(page, object):

        if hasattr(page, 'reload'):

            page.reload()

            return True

        else:

            return False

    else:

        raise TypeError("Input must be an object")

