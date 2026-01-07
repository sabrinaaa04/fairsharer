import numpy as np
from fairsharer.fair_sharer import fair_sharer


def test_fair_sharer_examples():
    values = [0, 1000, 800, 0]

    result_1 = fair_sharer(values, 1)
    assert np.allclose(result_1, [100, 800, 900, 0])

    result_2 = fair_sharer(values, 2)
    assert np.allclose(result_2, [100, 890, 720, 90])
