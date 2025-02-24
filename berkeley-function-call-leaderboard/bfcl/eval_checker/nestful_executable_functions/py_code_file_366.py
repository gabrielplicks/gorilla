from typing import Any

from unittest.mock import Mock



def always_return_seven() -> int:

    """

    A function that always returns the integer 7.

    This function includes a mock function call to generate mock data for testing purposes.

    Args:

        None

    Returns:

        The integer 7.

    """

    mock_function = Mock()

    mock_function.return_value = 7

    return mock_function()

