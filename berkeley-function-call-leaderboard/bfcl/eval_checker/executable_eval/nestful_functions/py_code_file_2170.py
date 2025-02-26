import urllib.parse



def convert_dict_to_url_encoding(dictionary: dict[str, str]) -> str:

    """Converts a given dictionary to a url-encoded string.



    Args:

        dictionary: A mapping from string keys to string values.



    Returns:

        A string of all key-value pairs url-encoded.

    """

    url_encoded_string = ''

    for key, value in dictionary.items():

        url_encoded_key = urllib.parse.quote_plus(key)

        url_encoded_value = urllib.parse.quote_plus(value)

        url_encoded_string += url_encoded_key + '=' + url_encoded_value + '&'

    return url_encoded_string[:-1]  # Remove the last '&' character

