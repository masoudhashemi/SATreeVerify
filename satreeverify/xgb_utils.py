import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from z3 import *

XMIN = -np.inf
XMAX = np.inf


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


def insert_val(l, index, value, fill=-1):
    """ helper function to insert a value into a list
        at the given index.

    :param l: list to be filled
    :param index: where to insert the value
    :param value: value to be inserted
    :param fill: fill the gaps with this value
    """
    if len(l) > index:
        l[index] = value
    else:
        while (len(l)) < index:
            l.append(fill)
        l.append(value)


def box_intersection(mini, minj, maxi, maxj):
    maxm = np.maximum(mini[:, :, None], minj.T[None, :, :])
    minm = np.minimum(maxi[:, :, None], maxj.T[None, :, :])
    # It looks like min > max, but these are for u, l
    check = np.all(((minm - maxm) > 0), axis=1)

    return check


def get_ens_thresh(dump):
    """Finds all thresholds in xgboost.
    (from FactorForest)
    """
    splits = []

    def _dfs(tree):
        if "leaf" not in tree:

            nodeid, children = tree["nodeid"], tree["children"]
            split, threshold = tree["split"], float(tree["split_condition"])

            if split[0] == "f":
                split = int(split[1:])
            else:
                split = int(split)

            splits.append((split, threshold))

            # TODO: handle "missing" values
            node0, node1 = children[0]["nodeid"], children[1]["nodeid"]
            if (node0 == tree["yes"]) and (node1 == tree["no"]):
                left_subtree = children[0]
                right_subtree = children[1]
            elif (node1 == tree["yes"]) and (node0 == tree["no"]):
                left_subtree = children[1]
                right_subtree = children[0]
            else:
                raise ValueError("node ids do not match!")

            _dfs(left_subtree)
            _dfs(right_subtree)

    for i in range(len(dump)):
        tree = json.loads(dump[i])
        _dfs(tree)

    return list(set(splits))


def leaf_boxes(tree, num_feats):
    """Finds the boundaries and values of the leaves in xgboost.
    (from FactorForest)
    """
    minms, maxms, vals, nodeids = [], [], [], []
    box_min = XMIN * np.ones(num_feats)
    box_max = XMAX * np.ones(num_feats)

    def _dfs(tree, box_min, box_max):
        if "leaf" in tree:
            minms.append(box_min)
            maxms.append(box_max)
            vals.append(tree["leaf"])
            nodeids.append(tree["nodeid"])
        else:
            nodeid, children = tree["nodeid"], tree["children"]
            split, threshold = tree["split"], tree["split_condition"]

            if split[0] == "f":
                split = int(split[1:])
            else:
                split = int(split)

            # TODO: handle "missing" values
            node0, node1 = children[0]["nodeid"], children[1]["nodeid"]
            if (node0 == tree["yes"]) and (node1 == tree["no"]):
                left_subtree = children[0]
                right_subtree = children[1]
            elif (node1 == tree["yes"]) and (node0 == tree["no"]):
                left_subtree = children[1]
                right_subtree = children[0]
            else:
                raise ValueError("node ids do not match!")

            lower, upper = box_min[split], box_max[split]

            # TODO: find an alternative that does not require so much copying
            left_box_min = np.copy(box_min)
            right_box_min = np.copy(box_min)

            left_box_max = np.copy(box_max)
            right_box_max = np.copy(box_max)

            left_box_min[split] = lower
            left_box_max[split] = threshold

            right_box_min[split] = threshold
            right_box_max[split] = upper

            _dfs(left_subtree, left_box_min, left_box_max)
            _dfs(right_subtree, right_box_min, right_box_max)

    _dfs(tree, box_min, box_max)

    nodeids = np.array(nodeids)
    order = np.argsort(nodeids)

    minms, maxms, vals = np.array(minms), np.array(maxms), np.array(vals)

    return nodeids[order], minms[order], maxms[order], vals[order]


def get_lineage_(left, right, features, threshold, value):
    """Helper function for lineage calculation."""

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = "l"
        elif child in right:
            parent = np.where(right == child)[0].item()
            split = "r"

        lineage.append([parent, split, threshold[parent], features[parent]])

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]
    all_paths = []
    for ic, child in enumerate(idx):
        if child == [0]:
            all_paths.append([0, value[child]])
        elif (child in right) or (child in left):
            path = []
            for node in recurse(left, right, child):
                path.append(node)
            path.append(value[child])
            all_paths.append(path)

    return all_paths


def get_tree_parts(branches):
    """Helper function to find the list of branches, leaves, and values of leaves."""
    left = []
    right = []
    feature = []
    threshold = []
    value = []
    for li in branches:
        if not isinstance(li, tuple):
            insert_val(left, li[0], li[-2])
            insert_val(right, li[0], li[-1])
            insert_val(feature, li[0], li[-3])
            insert_val(threshold, li[0], li[-4])
            insert_val(value, li[0], None)
        else:
            insert_val(left, li[0], -1)
            insert_val(right, li[0], -1)
            insert_val(feature, li[0], -2)
            insert_val(threshold, li[0], -1)
            insert_val(value, li[0], li[1])

    return (
        np.asarray(left),
        np.asarray(right),
        np.asarray(feature),
        np.asarray(threshold),
        np.asarray(value),
    )


def get_branches(tree):
    """ Find branches of the xgboost. """
    lineage = []

    def _dfs(tree, lineage=lineage, split=None):
        if "leaf" in tree:
            lineage.append((tree["nodeid"], tree["leaf"]))
        else:
            parent, children = tree["nodeid"], tree["children"]
            feature, threshold = tree["split"], float(tree["split_condition"])
            if feature[0] == "f":
                feature = int(feature[1:])
            else:
                feature = int(feature)

            node0, node1 = children[0]["nodeid"], children[1]["nodeid"]
            if (node0 == tree["yes"]) and (node1 == tree["no"]):
                left_subtree = children[0]
                right_subtree = children[1]
            elif (node1 == tree["yes"]) and (node0 == tree["no"]):
                left_subtree = children[1]
                right_subtree = children[0]
            else:
                raise ValueError("node ids do not match!")

            lineage.append(
                [parent, split, threshold, feature, tree["yes"], tree["no"]]
            )
            _dfs(left_subtree, lineage, split="l")
            _dfs(right_subtree, lineage, split="r")

    _dfs(tree, lineage, None)

    return lineage


def get_lineage(tree):
    """Creates the lineage (all paths) of xgboost."""
    branches = get_branches(tree)
    left, right, features, threshold, value = get_tree_parts(branches)
    all_paths = get_lineage_(left, right, features, threshold, value)
    return all_paths


def find_boundaries(dt):
    """Find the leaves boundaries."""
    lineages = get_lineage(dt)
    box_dict = {}
    for lineage in lineages:
        box_fr = list(set([li[-1] for li in lineage[:-2]]))
        box = {str(bi): [-np.inf, np.inf] for bi in box_fr}
        for li in lineage[:-2]:
            if li[1] == "l":
                box[str(li[-1])][1] = min(box[str(li[-1])][1], li[2])
            else:
                box[str(li[-1])][0] = max(box[str(li[-1])][0], li[2])
        box["value"] = lineage[-1]
        box_dict[f"{str(lineage[-2])}"] = box

    return box_dict


def create_leaf_regions(dict_boundaries, var_x, tree_num):
    """ Creates propositional formulation of the leaves boundaries.

    :param dict_boundaries: a dictionary of the boundaries for each leaf
        `rfutils.find_boundaries` and `xgbutils.find_boundaries` can be
        used to create the dictionary.
    :param var_x: list of the z3 bool variables associated with discrete
        data.
    :param tree_num: index of the tree in ensemble.
    :return: list of the regions and a list of created constraints.
    """
    all_c = []
    constraints = []
    for c_node, x_val_dict in dict_boundaries.items():
        c_name = f"c({tree_num},{c_node})"
        current_c = z3.Bool(c_name)
        if c_node != "0":
            all_c.append(current_c)
        else:
            print(f"tree{tree_num}: root node is a leaf.")
        cons = []
        for k, v in x_val_dict.items():
            if k != "value":
                k_name_0 = f"{k}_{v[0]}"
                k_name_1 = f"{k}_{v[1]}"

                if np.isinf(v[0]):
                    xi1 = var_x[f"x({k_name_1})"]
                    cons.append(z3.Not(xi1))
                elif np.isinf(v[1]):
                    xi0 = var_x[f"x({k_name_0})"]
                    cons.append(xi0)
                else:
                    xi0 = var_x[f"x({k_name_0})"]
                    xi1 = var_x[f"x({k_name_1})"]
                    cons.append(z3.Not(xi1))
                    cons.append(xi0)
            else:
                constraints.append((current_c, v))
        if len(cons) > 0:
            constraints.append(current_c == z3.And(*cons))

    if len(all_c) > 1:
        # at least one of the leaves of each tree should be one
        # at most one of the leaves of the tree should be one.
        constraints.append(z3.Or(all_c))
        for ci in all_c:
            other_c = set(all_c).difference({ci})
            constraints.append(z3.Implies(ci, z3.Not(z3.Or(other_c))))
    return all_c, constraints


def get_output(opt, c_weights):
    opt_vlues = {str(v): opt.model()[v] for v in opt.model() if "c" in str(v)}
    adv_weights = {
        ci: c_weights[ci]
        for ci, vi in opt_vlues.items()
        if vi and ci in c_weights.keys()
    }
    return adv_weights


def get_varx_i(var_x, ind):
    """Helper function to get z3 variables about the feature [ind]"""
    xi = {
        v: float(k[2:-1].split("_")[1])
        for k, v in var_x.items()
        if k[2:].split("_")[0] == str(ind)
    }
    return xi


def linf_const_x(var_x, sample, epsilon):
    """Constraints on the input features associated with infinity box.
    x value cannot be out of the infinity norm ball.
    """
    mini, maxi = linf_sample(sample, epsilon)
    linf_cons = []
    for i in range(sample.shape[1]):
        xi = get_varx_i(var_x, i)
        if len(xi) > 0:
            for k, v in xi.items():
                if v <= mini[0, i]:
                    linf_cons.append(k)
                elif v > maxi[0, i]:
                    linf_cons.append(z3.Not(k))
    return linf_cons


def create_x_conditions(var_x, all_thresh):
    """ Creates constraints on value relations.
    If x[0]>10 ==> x(0_10)=True then x[0]>8 ==> x(0_8)=True.
    If x[1]<4 ==> x(1_40)=False then x[1]<10 ==> x(1_10)=False.
    The constraints make sure this relations are forced in propositional
    formulation.
    """
    feature_nums = set([v[0] for v in all_thresh])
    x_conds = []
    for fi in feature_nums:
        thresh_list = sorted(
            list(set([v[1] for v in all_thresh if v[0] == fi]))
        )
        for thi in range(1, len(thresh_list)):
            x_conds.append(
                z3.Implies(
                    var_x[f"x({fi}_{thresh_list[thi]})"],
                    var_x[f"x({fi}_{thresh_list[thi - 1]})"],
                )
            )

            x_conds.append(
                z3.Implies(
                    z3.Not(var_x[f"x({fi}_{thresh_list[thi - 1]})"]),
                    z3.Not(var_x[f"x({fi}_{thresh_list[thi]})"]),
                )
            )

    return x_conds


def linf_sample(sample, epsilon):
    """ Creates boundaries of the infinity norm ball around the sample.

    :param sample: the sample to create the ball for.
    :param epsilon: size of the ball in ratio format.
    :return: list of mins and list of max
    """
    maxi = []
    mini = []
    for i in range(sample.shape[1]):
        if sample[0, i] != 0:
            maxi.append(
                np.max(
                    [
                        sample[0, i] * (1 - epsilon),
                        sample[0, i] * (1 + epsilon),
                    ]
                )
            )
            mini.append(
                np.min(
                    [
                        sample[0, i] * (1 - epsilon),
                        sample[0, i] * (1 + epsilon),
                    ]
                )
            )
        else:
            maxi.append(epsilon)
            mini.append(-epsilon)

    mini = np.asarray(mini).reshape(1, -1)
    maxi = np.asarray(maxi).reshape(1, -1)
    return mini, maxi


def create_all_smt(clf, var_x, sample, epsilon, lower_bound=False):
    predict = clf.predict(sample)[0]
    print(predict)

    _dump = clf.get_booster().get_dump(dump_format="json")
    c_weights = {}
    orig_weights = {}
    all_c = []
    remove_c = []

    linf_conds = linf_const_x(var_x, sample, epsilon)
    all_thresh = get_ens_thresh(_dump)
    x_conditions = create_x_conditions(var_x, all_thresh)

    core_constraints = set()
    soft_constraints = set()

    for ci in linf_conds:
        core_constraints.add(ci)
    for ci in x_conditions:
        core_constraints.add(ci)

    mini, maxi = linf_sample(sample, epsilon)

    leaf_apply = clf.apply(sample)[0]
    for i, estimator in enumerate(_dump):

        leaf = leaf_apply[i]
        remove_c.append(f"c({i},{leaf})")

        estimator = json.loads(estimator)
        branches = get_branches(estimator)
        (left, right, feature, _, value,) = get_tree_parts(branches)

        (leaves, min_corners, max_corners, _,) = leaf_boxes(
            estimator, clf._features_count
        )

        intersect_i = box_intersection(min_corners, mini, max_corners, maxi)

        orig_weights[f"c({i},{leaf})"] = value[leaf]

        leaves = np.where(feature == -2)[0]
        for ci in leaves[np.reshape(intersect_i, -1)]:
            all_c.append(f"c({i},{ci})")

    # if there is no overlapping leaf, terminate the run
    remaining_c = set(all_c).difference(remove_c)

    not_cs = []
    for i, estimator in enumerate(_dump):
        estimator = json.loads(estimator)
        dict_boundaries = find_boundaries(estimator)
        _, constraints = create_leaf_regions(dict_boundaries, var_x, i)

        list_c = [ci for ci in constraints if isinstance(ci, tuple)]
        for ci in list_c:
            c_weights[str(ci[0])] = ci[1]

        for ci in constraints:
            if isinstance(ci, tuple):
                if f"{ci[0]}" in all_c:
                    if f"{ci[0]}" not in remove_c:
                        if predict:
                            soft_constraints.add((ci[0], (-ci[1])))
                        else:
                            soft_constraints.add((ci[0], (ci[1])))
                    else:
                        not_cs.append(ci[0])
                else:
                    core_constraints.add(Not(ci[0]))
            else:
                core_constraints.add(ci)

    if not lower_bound:
        core_constraints.add(Not(And(not_cs)))

    return (
        list(core_constraints),
        list(soft_constraints),
        c_weights,
        orig_weights,
        all_c,
    )


def binerize_val(val, nbits, nbitsnew, w):

    wmin_ = w.min()
    wmax_ = w.max()

    wmin = -1 * max(-wmin_, wmax_)
    wmax = max(-wmin_, wmax_)

    if val > wmax:
        val = wmax
    elif val < wmin:
        val = wmin

    val_ = np.abs(val) / wmax
    sgn_ = np.sign(val) if val != 0 else 1

    max_val = 2 ** (nbits) - 1

    val_ = int(val_ * max_val)
    val_b = [int(bi) for bi in "{0:b}".format(val_)]
    val_ = copy.deepcopy(val_b)

    while len(val_) < nbitsnew:
        val_.insert(0, 0)

    v = [bool(vi) for vi in val_]

    if sgn_ == 1:
        v.insert(0, False)
    else:
        v.insert(0, True)

    return v


def list_c_val(c_weights, nbits, nbitsnew):
    list_c_ = []
    list_val_ = []

    w = np.asarray([v for k, v in c_weights.items()])

    for k, v in c_weights.items():
        list_val_.append(binerize_val(v, nbits, nbitsnew, w))
        list_c_.append(Bool(k))

    return list_val_, list_c_


def sum_loop_pos(xin, c_in, n):

    x_ = []
    c_ = []
    for ci, xi in zip(c_in, xin):
        if not xi[0]:
            x_.append(xi[1:])
            c_.append(ci)

    constraints = set()
    d = {}
    xh = {}
    for i in range(1, n + 1):
        d[(0, i)] = Bool(f"dsump_{0}({i})")
        xh[(0, i)] = Bool(f"xhsump_{0}({i})")

    if len(x_) == 0:
        for i in range(1, n + 1):
            constraints.add(Not(d[(0, i)]))
        return constraints, 0

    if len(x_) == 1:
        for i in range(1, n + 1):
            cons = xh[(0, i)] if x_[0][i - 1] else Not(xh[(0, i)])
            constraints.add(cons)
            constraints.add(d[(0, i)] == And(c_[0], xh[(0, i)]))
        return constraints, 0

    c = {}
    x = {}
    for seq_num in range(1, len(x_)):

        c[(seq_num, 0)] = Bool(f"csump_{seq_num}({0})")
        for i in range(1, n + 1):
            d[(seq_num, i)] = Bool(f"dsump_{seq_num}({i})")
            c[(seq_num, i)] = Bool(f"csump_{seq_num}({i})")
            x[(seq_num, i)] = Bool(f"xsump_{seq_num}({i})")
            xh[(seq_num, i)] = Bool(f"xhsump_{seq_num}({i})")

    for seq_num in range(1, len(x_)):

        constraints.add((Not(c[(seq_num, n)])))

        if seq_num == 1:
            for i in range(1, n + 1):
                cons = xh[(0, i)] if x_[0][i - 1] else Not(xh[(0, i)])
                constraints.add(cons)
                constraints.add(d[(0, i)] == And(c_[0], xh[(0, i)]))

        for i in range(1, n + 1):
            cons = (
                xh[(seq_num, i)]
                if x_[seq_num][i - 1]
                else Not(xh[(seq_num, i)])
            )
            constraints.add(cons)
            constraints.add(
                x[(seq_num, i)] == And(c_[seq_num], xh[(seq_num, i)])
            )

            constraints.add(
                (
                    c[(seq_num, i - 1)]
                    == Or(
                        And(x[(seq_num, i)], d[(seq_num - 1, i)]),
                        And(x[(seq_num, i)], c[(seq_num, i)]),
                        And(c[(seq_num, i)], d[(seq_num - 1, i)]),
                    )
                )
            )

            constraints.add(
                (
                    d[(seq_num, i)]
                    == (
                        x[(seq_num, i)]
                        == (d[(seq_num - 1, i)] == c[(seq_num, i)])
                    )
                )
            )
    return constraints, len(x_) - 1


def sum_loop_neg(xin, c_in, n):

    x_ = []
    c_ = []
    for ci, xi in zip(c_in, xin):
        if xi[0]:
            x_.append(xi[1:])
            c_.append(ci)

    constraints = set()
    d = {}
    xh = {}
    for i in range(1, n + 1):
        d[(0, i)] = Bool(f"dsumn_{0}({i})")
        xh[(0, i)] = Bool(f"xhsumn_{0}({i})")

    if len(x_) == 0:
        for i in range(1, n + 1):
            constraints.add(Not(d[(0, i)]))
        return constraints, 0

    if len(x_) == 1:
        for i in range(1, n + 1):
            cons = xh[(0, i)] if x_[0][i - 1] else Not(xh[(0, i)])
            constraints.add(cons)
            constraints.add(d[(0, i)] == And(c_[0], xh[(0, i)]))
        return constraints, 0

    c = {}
    x = {}

    for seq_num in range(1, len(x_)):

        c[(seq_num, 0)] = Bool(f"csumn_{seq_num}({0})")
        for i in range(1, n + 1):
            d[(seq_num, i)] = Bool(f"dsumn_{seq_num}({i})")
            c[(seq_num, i)] = Bool(f"csumn_{seq_num}({i})")
            x[(seq_num, i)] = Bool(f"xsumn_{seq_num}({i})")
            xh[(seq_num, i)] = Bool(f"xhsumn_{seq_num}({i})")

    for seq_num in range(1, len(x_)):

        constraints.add((Not(c[(seq_num, n)])))

        if seq_num == 1:
            for i in range(1, n + 1):
                cons = xh[(0, i)] if x_[0][i - 1] else Not(xh[(0, i)])
                constraints.add(cons)
                constraints.add(d[(0, i)] == And(c_[0], xh[(0, i)]))

        for i in range(1, n + 1):
            cons = (
                xh[(seq_num, i)]
                if x_[seq_num][i - 1]
                else Not(xh[(seq_num, i)])
            )
            constraints.add(cons)
            constraints.add(
                x[(seq_num, i)] == And(c_[seq_num], xh[(seq_num, i)])
            )

            constraints.add(
                (
                    c[(seq_num, i - 1)]
                    == Or(
                        And(x[(seq_num, i)], d[(seq_num - 1, i)]),
                        And(x[(seq_num, i)], c[(seq_num, i)]),
                        And(c[(seq_num, i)], d[(seq_num - 1, i)]),
                    )
                )
            )

            constraints.add(
                (
                    d[(seq_num, i)]
                    == (
                        x[(seq_num, i)]
                        == (d[(seq_num - 1, i)] == c[(seq_num, i)])
                    )
                )
            )
    return constraints, len(x_) - 1


def const_larger(seq_nump, seq_numn, n):

    const = []
    const_or = [
        And(Bool(f"dsump_{seq_nump}(1)"), Not(Bool(f"dsumn_{seq_nump}(1)")))
    ]
    for i in range(1, n):
        const_and = []
        for j in range(i):
            const_and.append(
                Bool(f"dsumn_{seq_numn}({j + 1})")
                == Bool(f"dsump_{seq_nump}({j + 1})")
            )
        const_and.append(Bool(f"dsump_{seq_nump}({i + 1})"))
        const_and.append(Not(Bool(f"dsumn_{seq_numn}({i + 1})")))

        const_or.append(And(const_and))

    const.append(Bool("class") == Or(const_or))

    return const


def soft_attack(clf, sample, epsilon, var_x):
    (
        core_constraints,
        soft_constraints,
        c_weights,
        orig_weights,
        all_c,
    ) = create_all_smt(clf, var_x, sample, epsilon)
    s = Optimize()
    s.set("timeout", 5000)
    for ci in core_constraints:
        s.add(ci)
    for ci in soft_constraints:
        s.add_soft(ci[0], ci[1])

    print(s.check())
    return s, c_weights


def hard_attack(clf, sample, epsilon, var_x, nbits):
    (
        core_constraints,
        soft_constraints,
        c_weights,
        orig_weights,
        all_c,
    ) = create_all_smt(clf, var_x, sample, epsilon)

    dump = clf.get_booster().get_dump(dump_format="json")
    ntrees = len(dump)
    new_nbits = int(np.ceil(np.log2(ntrees)) + nbits)

    list_val_, list_c_ = list_c_val(c_weights, nbits, new_nbits)
    sum_constp, seq_nump = sum_loop_pos(list_val_, list_c_, new_nbits)
    sum_constn, seq_numn = sum_loop_neg(list_val_, list_c_, new_nbits)

    w = np.asarray([v for k, v in c_weights.items()])
    const_class = const_larger(seq_nump, seq_numn, new_nbits)

    s = Solver()

    s.set("timeout", 100000)
    for ci in core_constraints:
        s.add(ci)

    for ci in sum_constp:
        s.add(ci)

    for ci in sum_constn:
        s.add(ci)

    for ci in const_class:
        s.add(ci)

    s.add([Not(Bool("class")) if clf.predict(sample)[0] else Bool("class")])

    print(s.check())

    return s, c_weights, seq_nump, seq_numn


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
