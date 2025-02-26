import re

from typing import List



def redact_document(document: str, sensitive_words: List[str]) -> str:

    """

    Returns a new string where the given sensitive words in the document are replaced

    with redacted text (e.g., 'XXXXXX').



    Args:

        document: The document to be redacted.

        sensitive_words: A list of sensitive words to be redacted.

    """

    regexes = [re.compile(word, re.IGNORECASE) for word in sensitive_words]

    for regex in regexes:

        document = regex.sub('XXXXXX', document)



    return document

