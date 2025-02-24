from typing import Dict, Any, Union



def convert_action_1(action: Dict[str, Any]) -> list:

    """Converts an action from a dictionary with 'x' and 'y' keys to a list with two elements.



    Args:

        action: A dictionary with 'x' and 'y' keys.



    Returns:

        A list with two elements.

    """

    return [action['x'], action['y']]



def convert_action_2(action: Union[Dict[str, Any], object]) -> list:

    """Converts an action from an object with 'x' and 'y' attributes to a list with two elements.



    Args:

        action: An object with 'x' and 'y' attributes.



    Returns:

        A list with two elements.

    """

    return [action.x, action.y]

