from typing import Union



def sentiment_progress_bar(polarity: Union[int, float]) -> str:

    """

    Returns a color ("green", "red", or "gray") based on the polarity score.

    Args:

        polarity: The polarity score.

    """

    if polarity > 0:

        return "green"

    elif polarity < 0:

        return "red"

    else:

        return "gray"

