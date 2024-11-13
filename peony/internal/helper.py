import numpy


def should_run(prob: float) -> bool:
    if prob == 0.0:
        return False
    if prob == 1.0:
        return True

    return np.random.random() < prob
