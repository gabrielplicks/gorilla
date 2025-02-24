from typing import List



def import_as_dict(local_variables: List[str]) -> dict:

    """Creates a dictionary from a list of local variables.



    Args:

        local_variables: A list of local variables.



    Returns:

        A dictionary where the keys and values are the local variables.

    """

    assert len(local_variables) > 0, "Local variables are empty"

    return dict(zip(local_variables, local_variables))

