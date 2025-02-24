import re

from typing import List



def tokenize_with_start_end_tokens(sentence: str, start_token: str, end_token: str, punctuations: List[str]) -> List[str]:

    """Tokenizes a sentence with start and end tokens, handling punctuation.



    Args:

        sentence: The input sentence.

        start_token: The start token.

        end_token: The end token.

        punctuations: A list of punctuation marks to ignore while tokenization.



    """

    sentence = re.sub(r"([{}])".format("".join(punctuations)), r" \1 ", sentence)

    tokens = [start_token] + sentence.split() + [end_token]



    return tokens

