import base64



def decode_and_reencode_base64(base64_encoded_str: str) -> str:

    """

    Decodes a Base64-encoded string and then re-encodes it back into Base64.

    Args:

        base64_encoded_str: The Base64-encoded string to decode and re-encode.

    """

    try:

        decoded_str = base64.b64decode(base64_encoded_str).decode("utf-8")

        reencoded_str = base64.b64encode(decoded_str.encode("utf-8")).decode("utf-8")

        return reencoded_str

    except Exception as e:

        return f"Error: {e}"

