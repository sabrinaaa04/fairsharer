import numpy as np

def fair_sharer(values, num_iterations, share=0.1):
    """
    Runs num_iterations.
    In each iteration the highest value in "values" gives a fraction (share)
    to both the left and right neighbor. The leftmost field is considered
    the neighbor of the rightmost field.

    Examples:
    fair_sharer([0, 1000, 800, 0], 1) --> [100, 800, 900, 0]
    fair_sharer([0, 1000, 800, 0], 2) --> [100, 890, 720, 90]

    Args:
        values: list or numpy array
        num_iterations: number of iterations
        share: fraction given to each neighbor
    """
    values_new = np.array(values, dtype=float)
    n = len(values_new)

    for _ in range(num_iterations):
        max_index = np.argmax(values_new)
        max_value = values_new[max_index]

        amount = max_value * share
        left = (max_index - 1) % n
        right = (max_index + 1) % n

        values_new[max_index] -= 2 * amount
        values_new[left] += amount
        values_new[right] += amount

    return values_new
