import json

from typing import Dict



def update_user_settings(settings_file: str, user_settings: Dict[str, str]) -> Dict[str, Dict[str, str]]:

    """Updates a dictionary of user settings by loading them from a JSON file and merging them with the given settings.



    Args:

        settings_file: The path to the JSON file containing the user settings.

        user_settings: A dictionary of user settings to merge with the settings from the file.



    Returns:

        The updated dictionary of user settings.

    """

    with open(settings_file, 'r') as f:

        file_settings = json.load(f)



    file_settings['settings'].update(user_settings)

    return file_settings

