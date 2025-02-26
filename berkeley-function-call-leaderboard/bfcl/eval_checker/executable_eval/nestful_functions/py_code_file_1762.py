from typing import AnyStr



def is_valid_excel_sheet_name(name: AnyStr) -> bool:

    """Validates the given Excel sheet name `name` is valid.



    The `name` must be between 1 and 30 characters and consist only of uppercase letters, lowercase letters, and digits.

    The first character must be a letter, and the rest can be letters or digits.



    Args:

        name: The name of the Excel sheet to validate.



    Returns:

        True if the name is valid, False otherwise.

    """

    if len(name) < 1 or len(name) > 30:

        return False



    if not name[0].isalpha():

        return False



    for char in name[1:]:

        if not char.isalnum():

            return False



    return True

