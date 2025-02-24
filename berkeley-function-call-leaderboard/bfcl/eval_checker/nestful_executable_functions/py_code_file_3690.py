import re



def get_name_from_address(address: str) -> str:

    """Returns a string that is either "Hi, John" if the address contains the name "John", "Hi, Jane" if the address contains the name "Jane", or "Hi there" if the address does not contain any of these names.



    Args:

        address: The address to search for the name.



    Returns:

        A string that is either "Hi, John" if the address contains the name "John", "Hi, Jane" if the address contains the name "Jane", or "Hi there" if the address does not contain any of these names.

    """

    name_pattern = r"(John|Jane)"

    match = re.search(name_pattern, address)

    if match:

        return f"Hi, {match.group()}"

    return "Hi there"

