import os

import typing



def check_env_var(env_var_name: str, expected_value: typing.Any) -> bool:

    """Checks if an environment variable is set and has a certain value.



    Args:

        env_var_name: The name of the environment variable.

        expected_value: The value to compare to.



    Returns:

        A boolean indicating whether the variable is set and has the correct value.

    """

    env_var_value = os.environ.get(env_var_name)

    if env_var_value is None:

        return False

    return env_var_value == expected_value

