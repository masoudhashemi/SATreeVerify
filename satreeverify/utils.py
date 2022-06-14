import numpy as np


def box_intersection(mini, minj, maxi, maxj):
    maxm = np.maximum(mini[:, :, None], minj.T[None, :, :])
    minm = np.minimum(maxi[:, :, None], maxj.T[None, :, :])
    # It looks like min > max, but these are for u, l
    check = np.all(((minm - maxm) > 0), axis=1)

    return check
