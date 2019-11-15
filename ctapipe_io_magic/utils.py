import numpy as np


def index_in_array(idx, arr):
    """Return indices of elements of `idx` in `arr`.

    Parameters:
        idx : np.array, indices to find
        arr : np.array, look in this array for indices
    """
    return np.asarray(np.nonzero(idx.reshape(-1, 1) == arr.reshape(1, -1)))[1, :]


def test_index_in_array():
    event_nums = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7])
    idx = np.array([1, 2, 3, 10, 20])

    result_idx_in = np.array([1, 2, 3])
    result = np.array([5, 1, 6])

    notin = np.isin(idx, event_nums, invert=True)
    isin = ~notin

    assert notin.sum() > 0
    assert isin.sum() == len(result_idx_in)

    idx_in = idx[isin]

    np.testing.assert_array_equal(idx_in, result_idx_in)

    index = index_in_array(idx, event_nums)

    assert np.isin(event_nums[index], idx, invert=True).sum() == 0

    np.testing.assert_array_equal(index, result)
    np.testing.assert_array_equal(event_nums[index], event_nums[result])
