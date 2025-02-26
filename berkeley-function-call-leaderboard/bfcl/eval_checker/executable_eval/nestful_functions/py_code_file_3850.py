import json

from typing import Dict



def clean_body_for_vcr(cassette: Dict) -> Dict:

    """Cleans the response body of a VCR cassette to make it replayable.



    This function removes headers, cookies, and other sensitive information from the response body

    to obfuscate the request and response, while preserving the essential structure of the data.



    Args:

        cassette: A dictionary representing a VCR cassette.



    Returns:

        The cleaned cassette dictionary.

    """

    for interaction in cassette['interactions']:

        for response in interaction['response']['body']['string']:

            try:

                response['body']['string'] = json.dumps(json.loads(response['body']['string']))

            except json.decoder.JSONDecodeError:

                pass

    return cassette

