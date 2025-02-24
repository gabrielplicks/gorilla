from typing import Optional



def get_header_value(header: str, key: str) -> Optional[str]:

    """

    Returns the value corresponding to the given `key` in the `header`.

    If the `key` does not exist in the `header`, returns `None`.



    Args:

        header: The header string.

        key: The key to search for in the header.

    """

    for line in header.splitlines():

        parts = line.split(':')

        if len(parts) != 2:

            continue  # Skip invalid lines

        field_name, value = parts

        if field_name.strip().lower() == key.lower():

            return value.strip()

    return None

