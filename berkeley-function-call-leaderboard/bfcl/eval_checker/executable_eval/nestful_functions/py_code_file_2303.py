from typing import Optional



def get_param_value(param_name: str, default_value: Optional[str] = None, boolean_flag: bool = False) -> Optional[str]:

    """

    Retrieves the value of a command-line parameter.

    If the `boolean_flag` is set to `True`, it returns the `default_value` if it is not `None`.

    Args:

        param_name: The name of the command-line parameter.

        default_value: The default value for the parameter. If `None`, no default value is available.

        boolean_flag: A boolean flag that, when set to `True`, will return the `default_value` if it is not `None`.

    """

    if boolean_flag and default_value is not None:

        return default_value

