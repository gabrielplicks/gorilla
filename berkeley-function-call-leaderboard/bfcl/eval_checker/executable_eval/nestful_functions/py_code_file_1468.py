from typing import Dict



def parse_definitions(definitions: Dict[str, Dict[str, str]]) -> Dict[str, str]:

    """Parses and returns a dictionary of word definitions from a pickled dataset.



    Args:

        definitions: A dictionary with the following structure:

            {

                "word": {

                    "def": "definition",

                    "example": "example sentence",

                    "source": "source of definition"

                },

                ...

            }



    Returns:

        A dictionary with word definitions.

    """

    assert isinstance(definitions, dict) and all(

        'def' in word and isinstance(word['def'], str)

        for word in definitions.values()

    ), "Input must be a dictionary with word definitions"

    return {word: definitions[word]['def'] for word in definitions}

