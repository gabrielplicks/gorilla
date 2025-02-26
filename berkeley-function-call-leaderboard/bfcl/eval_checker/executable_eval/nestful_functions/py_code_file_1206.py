from typing import Union



def classify_prediction(prediction: Union[int, float]) -> str:

    """Classifies a prediction based on its value.



    Args:

        prediction: A number between 0 and 1.0 representing a prediction.



    Returns:

        A string indicating whether the prediction is successful, possible, or failed.

    """

    if prediction > 0.7:

        return 'Success!'

    elif 0.3 <= prediction <= 0.7:

        return 'Possible.'

    else:

        return 'Failed.'

