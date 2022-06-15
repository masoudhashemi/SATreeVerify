import numpy as np
import pandas as pd
from z3 import Bool


def box_intersection(mini, minj, maxi, maxj):
    maxm = np.maximum(mini[:, :, None], minj.T[None, :, :])
    minm = np.minimum(maxi[:, :, None], maxj.T[None, :, :])
    # It looks like min > max, but these are for u, l
    check = np.all(((minm - maxm) > 0), axis=1)

    return check


def get_varx_i(var_x, ind):
    """Helper function to get z3 variables about the feature [ind]"""
    xi = {
        v: float(k[2:-1].split("_")[1])
        for k, v in var_x.items()
        if k[2:].split("_")[0] == str(ind)
    }
    return xi


def disc_data(data, all_thresh):
    """ Discretize the data

    :param data: data to be disceretized.
    :param all_thresh: list of all thresholds in the trees
        `get_ens_thresh` can be used to find the thresholds.
    :return: :class:`pandas.DataFrame` with columns being the
        feature_threshold pairs and bool values. True meaning
        the value is larger than the threshold and False
        otherwise.
    """
    col_names = [f"{v[0]}_{v[1]}" for v in all_thresh]
    data_ = pd.DataFrame(columns=col_names)
    for v in all_thresh:
        col = f"{v[0]}_{v[1]}"
        data_[col] = data[:, v[0]] > v[1]
    return data_


def create_var_x(all_threshs):
    columns = [f"{v[0]}_{v[1]}" for v in all_threshs]
    var_x = {f"x({i})": Bool(f"x({i})") for i in columns}
    return var_x


def get_x_adv(solver, var_x, sample):
    x_adv = pd.DataFrame(
        ({k[2:-1]: [solver.model()[v]] for k, v in var_x.items()})
    )
    x_adv_sample, compare = create_adv_sample(x_adv, sample)
    return x_adv, x_adv_sample, compare


def create_adv_sample(x_adv, sample):
    """ Creates an adversarial example from the boolean z3 variables
    from the SAT solution.

    :param x_adv: adversarial example in bool format
    :param sample: the sample used to create the adversarial example
    :return: an adversarial example.
    """
    num_features = sample.shape[1]
    x_adv_num = np.zeros((1, num_features))
    columns = x_adv.columns

    bound_i = []

    for i in range(num_features):
        col_i = np.asarray(
            [ci for ci in columns if int(ci.split("_")[0]) == i]
        )
        col_i_false = col_i[np.where(x_adv[col_i].values == False)[1]]
        col_i_true = col_i[np.where(x_adv[col_i].values == True)[1]]

        if len(col_i_false) > 0 and len(col_i_true) > 0:
            max_i = np.asarray([float(ci.split("_")[1]) for ci in col_i_false])
            max_i = max_i.min()
            min_i = np.asarray([float(ci.split("_")[1]) for ci in col_i_true])
            min_i = min_i.max()
            # x_adv_num[0, i] = (max_i + min_i) / 2
            if sample[0, i] <= min_i:
                x_adv_num[0, i] = min_i + (max_i - min_i) * 0.001
            elif sample[0, i] >= max_i:
                x_adv_num[0, i] = max_i - (max_i - min_i) * 0.001
            else:
                x_adv_num[0, i] = sample[0, i]

        elif len(col_i_false) > 0 and len(col_i_true) == 0:
            min_i = -np.inf
            max_i = np.asarray([float(ci.split("_")[1]) for ci in col_i_false])
            max_i = max_i.min()
            if sample[0, i] <= max_i:
                x_adv_num[0, i] = sample[0, i]
            else:
                x_adv_num[0, i] = max_i * (1 - np.sign(max_i) * 0.001)

        elif len(col_i_false) == 0 and len(col_i_true) > 0:
            min_i = np.asarray([float(ci.split("_")[1]) for ci in col_i_true])
            min_i = min_i.max()
            max_i = np.inf
            if sample[0, i] >= min_i:
                x_adv_num[0, i] = sample[0, i]
            else:
                x_adv_num[0, i] = min_i * (1 + np.sign(min_i) * 0.001)

        else:
            min_i = -np.inf
            max_i = np.inf
            x_adv_num[0, i] = sample[0, i]

        bound_i.append([min_i, max_i])

    data_comp = np.r_[sample, x_adv_num]
    compare = pd.DataFrame(
        data=data_comp, columns=[str(i) for i in range(data_comp.shape[1])]
    ).T
    compare["diff(%)"] = np.abs(compare[1] - compare[0]) / np.abs(compare[0])
    compare["bound"] = bound_i

    return x_adv_num, compare
