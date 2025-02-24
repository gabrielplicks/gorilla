from typing import List



def process_text_list(texts: List[str], keywords: List[str]) -> List[str]:

    """Removes all detected keywords from a list of texts.



    Args:

        texts: A list of texts.

        keywords: A list of keywords.



    Returns:

        A new list of texts with all detected keywords removed.

    """

    processed_texts = []

    for text in texts:

        processed_text = " ".join([word for word in text.split() if word not in keywords])

        processed_texts.append(processed_text)

    return processed_texts

