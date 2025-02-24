from typing import Dict



def create_dict_from_named_args(**kwargs: Dict[str, any]) -> Dict[str, any]:

    """Creates a dictionary from named arguments.

    Args:

        kwargs: Any number of named arguments.

    """

    return dict(**kwargs)

