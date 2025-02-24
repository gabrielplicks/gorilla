from typing import AnyStr



def get_file_name_from_subject(subject: AnyStr) -> AnyStr:

    """

    Returns the file name from the subject string.

    The subject string is in the format of "[filename] - [file info]".

    The function splits the string by the dash and then removes any square brackets (`[]`) or parentheses (`()`) from the resulting string.



    Args:

        subject: The subject string in the format of "[filename] - [file info]".



    Returns:

        The file name from the subject string.

    """

    parts = subject.split(" - ")

    file_name = parts[0]

    file_name = file_name.replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    return file_name

