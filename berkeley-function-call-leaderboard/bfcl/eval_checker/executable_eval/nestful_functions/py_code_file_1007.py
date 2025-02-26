from typing import List



def convert_to_urls(string_list: List[str]) -> List[str]:

    """Converts a list of strings into URLs.



    Each string is assumed to be a file name with an extension. The function

    converts each string into a URL with the format

    'https://example.com/files/{file_type}/{file_name}.{file_type}'.



    Args:

        string_list: A list of strings.



    Returns:

        A list of URLs.

    """

    urls = []



    for string in string_list:

        file_name, file_type = string.split('.')

        url = f'https://example.com/files/{file_type}/{file_name}.{file_type}'

        urls.append(url)



    return urls

