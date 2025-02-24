from typing import List



def tokenize_sentence(sentence: str) -> List[str]:

    """Tokenizes a sentence by removing punctuation and converting words to lowercase.



    Args:

        sentence: The sentence to tokenize.



    Returns:

        A list of tokens.

    """

    punctuation = ",()!"

    lower_sentence = sentence.lower()

    for char in punctuation:

        lower_sentence = lower_sentence.replace(char, "")

    tokens = lower_sentence.split()

    return tokens



def tokenize_sentences(sentences: List[str]) -> List[str]:

    """Tokenizes a list of sentences by calling `tokenize_sentence` on each one.



    Args:

        sentences: The list of sentences to tokenize.



    Returns:

        A list of tokens.

    """

    tokens = []

    for sentence in sentences:

        tokens.extend(tokenize_sentence(sentence))

    return tokens

