from typing import Dict



def generate_policy(action: Dict[str, str]) -> Dict[str, str]:

    """Generates a policy from a single action.

    Args:

        action: A dictionary with a "name" key and a "value" key.

    Returns:

        A dictionary with the action name as the key and the action value as the value.

    """

    return {action["name"]: action["value"]}

