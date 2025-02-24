def simplify_string_replace(string: str, given_letter: str, replacement_letter: str) -> str:

    """Performs a simplified string replacement operation.

    Replaces all occurrences of the given letter in the string with the replacement letter.

    If the given letter doesn't exist in the string, the function simply returns the original string.



    Args:

        string: The input string.

        given_letter: The letter to be replaced.

        replacement_letter: The letter to replace the given letter with.

    """

    if given_letter not in string:

        return string



    replaced_string = ''

    for character in string:

        if character == given_letter:

            replaced_string += replacement_letter

        else:

            replaced_string += character



    return replaced_string

