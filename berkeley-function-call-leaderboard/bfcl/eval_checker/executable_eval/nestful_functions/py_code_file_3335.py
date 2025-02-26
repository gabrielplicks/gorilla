from typing import Union



def decode_string_or_bytes(input_value: Union[str, bytes], default_string: str = "") -> Union[str, None]:

    """Decodes a string or bytes object as a string if it's a bytes object.

    If it's already a string, it returns it without any modifications.

    Handles potential decoding errors by returning None or a default string.



    Args:

        input_value: The input value to be decoded.

        default_string: The default string to return in case of decoding errors.

    """

    if isinstance(input_value, str):

        return input_value



    try:

        decoded_string = input_value.decode("utf-8")

        return decoded_string

    except Exception:

        return default_string

