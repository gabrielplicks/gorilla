from typing import List, Dict



def radio_buttons_and_checks(button_list: List[Dict[str, str]]) -> str:

    """

    Generates an HTML string containing a set of radio buttons and checkboxes.



    Args:

        button_list: A list of dictionaries containing button definitions.

    """

    output = ""

    for button in button_list:

        button_type = button["type"]

        if button_type == "radio":

            checked = "checked" if button["checked"] else ""

            output += f'<input type="radio" name="{button["name"]}" value="{button["value"]}" {checked} />'

        elif button_type == "checkbox":

            checked = "checked" if button["checked"] else ""

            output += f'<input type="checkbox" name="{button["name"]}" value="{button["value"]}" {checked} />'

    return output

