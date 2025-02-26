from typing import List



def escape_special_chars_in_tokens(tokens: List[str]) -> List[str]:

    """Escapes special characters in tokens, except for the newline character `\n`, which is replaced with the line continuation character `\`.



    Args:

        tokens: A list of tokens.



    Returns:

        A list of tokens with special characters escaped.

    """

    escaped_tokens = []

    for token in tokens:

        escaped_token = ''

        for char in token:

            if char == '\n':

                escaped_token += '\\'

            elif char in ['\'', '"', '\\', '$', '(', ')', '*', '+', '.', '/', ':', '?', '[', ']', '{', '}', '=', '!', '<', '>']:

                escaped_token += '\\' + char

            else:

                escaped_token += char

        escaped_tokens.append(escaped_token)

    return escaped_tokens

