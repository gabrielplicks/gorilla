import urllib.parse



def generate_csv_url(filename: str, params: str) -> str:

    """Generates a valid CSV file URL based on the given file name and parameters.



    Args:

        filename: The name of the CSV file.

        params: The parameters to be included in the URL.



    Returns:

        A valid CSV file URL.

    """

    parsed_params = urllib.parse.parse_qs(params)

    query_string = urllib.parse.urlencode(parsed_params, doseq=True)

    url = f"https://hostname.com/api/download/csv/{filename}?params={query_string}"

    return url

