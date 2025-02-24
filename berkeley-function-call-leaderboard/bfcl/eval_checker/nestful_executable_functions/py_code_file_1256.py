from typing import Dict



def format_class_members(members: Dict[str, str]) -> str:

    """Formats class members into a string, excluding special members like `__str__`.



    Args:

        members: A dictionary of class members.

    """

    formatted_members = []



    for key, value in members.items():

        if key == '__str__':

            continue



        formatted_member = f'{key} = {value}'

        formatted_members.append(formatted_member)



    return ', '.join(formatted_members)

