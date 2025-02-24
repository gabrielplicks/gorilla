from typing import Dict, Union



def create_hash_string(d: Dict[str, str], values_as_int: bool = False) -> str:

    """Creates a unique hash string for a dictionary object with a fixed order.

    Args:

        d: The dictionary object containing only string keys and values.

        values_as_int: An optional Boolean parameter. If set to True, the values are converted to int.

    """

    assert isinstance(d, dict) and isinstance(values_as_int, bool)

    keys = sorted(d.keys())

    values = sorted(d.values())

    if values_as_int:

        values = [int(v) for v in values]

    hash_string = "_".join([str(k) for k in keys]) + "_" + "_".join([str(v) for v in values])



    return hash_string

