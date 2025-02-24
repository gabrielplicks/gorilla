from typing import Dict



def modify_params(params: Dict[str, any]) -> Dict[str, any]:

    """Modifies the keys and values of a dictionary.



    Args:

        params: The dictionary to be modified.



    Returns:

        A new dictionary with the modified keys and values.

    """

    modified_params = {

        "time_out": params.pop("timeout", None),

        "max_retry_count": params.pop("max_retries", None),

        "service_uri": params.pop("service_url", None),

        "enable_ssl": True if "enable_ssl" in params else None,

    }

    keys_to_remove = {k for k, v in params.items() if v in modified_params.values()}

    for key in keys_to_remove:

        params.pop(key, None)

    modified_params.update(params)



    return modified_params

