from typing import Dict, Any



def get_disabled_option(dictionary: Dict[str, Any]) -> str:

    """

    Checks if any of the options in the dictionary are not enabled, and returns the first such option.

    Args:

        dictionary: A dictionary containing options.

    Returns:

        The first option that is not enabled, or None if all options are enabled.

    """

    if "options" not in dictionary:

        return None



    options = dictionary["options"]

    if not options:  # Handle empty options dictionary

        return None



    for option, enabled in options.items():

        if not enabled:

            return option



    return None

