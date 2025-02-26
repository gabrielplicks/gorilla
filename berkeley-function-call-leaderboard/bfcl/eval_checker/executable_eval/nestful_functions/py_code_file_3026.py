import numpy as np



def compute_csi(truth: np.ndarray, forecast: np.ndarray) -> float:

    """Calculates the critical success index (CSI) given the truth and forecast arrays.



    Args:

        truth: A 2D numpy array containing binary values (0 or 1) indicating the presence or absence of a weather event.

        forecast: A 2D numpy array containing binary values (0 or 1) indicating the presence or absence of a weather event.

    """

    hit = np.logical_and(truth, forecast).sum()

    false_alarm = np.logical_and(forecast, np.logical_not(truth)).sum()

    true_negative = np.logical_and(np.logical_not(truth), np.logical_not(forecast)).sum()

    csi = hit / (hit + false_alarm + true_negative)

    return csi

