from typing import List



def clean_text(text: str, stopwords: List[str]) -> str:

    """

    Cleans a text by removing stopwords.



    Args:

        text: The input text.

        stopwords: A list of stopwords to remove from the text.



    Returns:

        The cleaned text.

    """

    words = text.split()  # Split the text into words

    cleaned_words = [word for word in words if word not in stopwords]  # Filter out the stopwords

    cleaned_text = ' '.join(cleaned_words)  # Join the remaining words together with spaces

    return cleaned_text

