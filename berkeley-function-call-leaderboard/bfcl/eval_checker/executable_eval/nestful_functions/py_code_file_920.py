from typing import List, Union



def split_on_word(sentence: Union[str, None], word: str) -> List[str]:

    """Splits a sentence on a specified word.

    Args:

        sentence: The sentence to split.

        word: The word to split the sentence on.

    """

    if sentence is None:

        return [sentence]

    if sentence == "":

        return []

    split_list = sentence.split(word)

    if len(split_list) > 1:

        return split_list

    return [sentence]

