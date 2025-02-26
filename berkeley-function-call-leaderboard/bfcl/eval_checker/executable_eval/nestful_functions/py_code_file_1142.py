from typing import Dict



def extract_and_manipulate_data(user_dict: Dict[str, str]) -> str:

    """Extracts and manipulates data from a dictionary containing user information.



    Args:

        user_dict: A dictionary containing user information.



    Returns:

        A string containing the user's name, email address, and phone number,

        concatenated into a single string. If the user's account is verified,

        the string "(verified)" is appended to the end of the concatenated string.

    """

    name = user_dict["name"]

    email = user_dict["email"]

    phone = user_dict["phone"]

    user_info = f"{name}, {email}"

    if "verified" in user_dict["account"] and user_dict["account"]["verified"]:

        user_info += " (verified)"



    return user_info

