import numpy as np
import pandas as pd
from sklearn.tree import _tree
from collections import OrderedDict
from z3 import *


def box_intersection(mini, minj, maxi, maxj):
    maxm = np.maximum(mini[:, :, None], minj.T[None, :, :])
    minm = np.minimum(maxi[:, :, None], maxj.T[None, :, :])
    # It looks like min > max, but these are for u, l
    check = np.all(((minm - maxm) > 0), axis=1)

    return check


def get_thresholds(dt):
    tree = dt.tree_
    dict_feat_thresh = {}
    for i in range(len(tree.feature)):
        if tree.feature[i] != _tree.TREE_UNDEFINED:
            if dict_feat_thresh.get(tree.feature[i], None) is None:
                dict_feat_thresh[tree.feature[i]] = [tree.threshold[i]]
            else:
                dict_feat_thresh[tree.feature[i]].append(tree.threshold[i])
    for k, v in dict_feat_thresh.items():
        dict_feat_thresh[k] = list(set(v))
    return dict_feat_thresh


def get_ens_thresh(clf):
    all_thresh = set()
    for dt in clf.estimators_:
        dict_thresh = get_thresholds(dt)
        for k, v in dict_thresh.items():
            for vi in v:
                all_thresh.add((k, vi))

    return list(all_thresh)


def disc_ens_data(data, all_thresh):
    col_names = [f"{v[0]}_{v[1]}" for v in all_thresh]
    data_ = pd.DataFrame(columns=col_names)
    for v in all_thresh:
        col = f"{v[0]}_{v[1]}"
        data_[col] = data[:, v[0]] > v[1]

    return data_


def path_to_leaves(children_left, children_right):
    """
    Given a tree, find the path between the root node and all the leaves
    """
    leaf_paths = OrderedDict()
    path = []

    def _find_leaves(root, path, depth, branch):
        children = [children_left[root], children_right[root]]
        children = [c for c in children if c != -1]

        if len(path) > depth:
            path[depth] = (root, branch)
        else:
            path.append((root, branch))

        if len(children) == 0:
            nodes, dirs = zip(*path[: depth + 1])
            leaf_paths[root] = list(zip(nodes[:-1], dirs[1:]))
        else:
            _find_leaves(children_left[root], path, depth + 1, "left")
            _find_leaves(children_right[root], path, depth + 1, "right")

    _find_leaves(0, path, 0, "root")
    return leaf_paths


def leaf_boxes(tree, x_min=-1e9, x_max=1e9):
    n_nodes = tree.node_count
    n_features = tree.n_features

    children_left = tree.children_left
    children_right = tree.children_right

    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    leaf_paths = path_to_leaves(children_left, children_right)
    leaves = sorted(leaf_paths)

    n_leaves = len(leaves)

    # Make this sparse in the future, if necessary
    max_corners = np.ones((n_leaves, n_features)) * x_max
    min_corners = np.ones((n_leaves, n_features)) * x_min

    for i, leaf in enumerate(leaves):
        path = leaf_paths[leaf]
        for node, step in path:
            feat_id = feature[node]
            thresh = threshold[node]

            # left, x <= t; r=t, l=-inf
            # right, x > t; r=inf, l=t

            if step == "left":
                max_corners[i, feat_id] = thresh
            else:
                min_corners[i, feat_id] = thresh

    return leaves, min_corners, max_corners


def build_rf_ensemble(model):
    num_trees = model.n_estimators

    forest_minms, forest_maxms, forest_vals = [], [], []
    all_leaves = []
    for i in range(num_trees):
        tree = model.estimators_[i].tree_
        leaves, minms, maxms = leaf_boxes(tree)
        vals = tree.value[leaves].squeeze()

        # The constant values stored in a random forest are counts of
        # the number of points for each class
        # We convert this into a probability for class 1
        vals = vals[:, 1] / vals.sum(-1)
        vals = (vals) / num_trees

        forest_minms.append(minms)
        forest_maxms.append(maxms)
        forest_vals.append(vals)

        all_leaves.append(leaves)

    return forest_minms, forest_maxms, forest_vals, all_leaves


def create_var_x(all_threshs):
    columns = [f"{v[0]}_{v[1]}" for v in all_threshs]
    var_x = {f"x({i})": Bool(f"x({i})") for i in columns}
    return var_x


def create_x_conditions(var_x, all_thresh):
    feature_nums = set([v[0] for v in all_thresh])
    x_conds = []
    for fi in feature_nums:
        thresh_list = sorted(list(set([v[1] for v in all_thresh if v[0] == fi])))
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


def get_varx_i(var_x, ind):
    xi = {
        v: float(k[2:-1].split("_")[1])
        for k, v in var_x.items()
        if k[2:].split("_")[0] == str(ind)
    }
    return xi


def linf_sample(sample, epsilon):
    maxi = []
    mini = []
    for i in range(sample.shape[1]):
        if sample[0, i] != 0:
            maxi.append(
                np.max([sample[0, i] * (1 - epsilon), sample[0, i] * (1 + epsilon)])
            )
            mini.append(
                np.min([sample[0, i] * (1 - epsilon), sample[0, i] * (1 + epsilon)])
            )
        else:
            maxi.append(epsilon)
            mini.append(-epsilon)

    mini = np.asarray(mini).reshape(1, -1)
    maxi = np.asarray(maxi).reshape(1, -1)
    return mini, maxi


def linf_const_x(var_x, sample, epsilon):
    mini, maxi = linf_sample(sample, epsilon)
    linf_cons = []
    for i in range(sample.shape[1]):
        xi = get_varx_i(var_x, i)
        if len(xi) > 0:
            for k, v in xi.items():
                if v < mini[0, i]:
                    linf_cons.append(k)
                elif v > maxi[0, i]:
                    linf_cons.append(z3.Not(k))
    return linf_cons


def get_lineage(tree):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = tree.tree_.feature
    tree_vals = tree.tree_.value.squeeze()
    values = tree_vals[:, 1] / tree_vals.sum(1)

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = "l"
        else:
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
        path = []
        for node in recurse(left, right, child):
            path.append(node)
        path.append(values[child])
        all_paths.append(path)

    return all_paths


def find_boundaries(dt):
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


def create_smt_regions(dict_boundaries, var_x, tree_num):
    all_c = []
    constraints = []
    for c_node, x_val_dict in dict_boundaries.items():
        c_name = f"c({tree_num},{c_node})"
        current_c = Bool(c_name)
        all_c.append(current_c)
        cons = []
        for k, v in x_val_dict.items():
            if k != "value":
                k_name_0 = f"{k}_{v[0]}"
                k_name_1 = f"{k}_{v[1]}"

                if np.isinf(v[0]):
                    xi1 = var_x[f"x({k_name_1})"]
                    cons.append(Not(xi1))
                elif np.isinf(v[1]):
                    xi0 = var_x[f"x({k_name_0})"]
                    cons.append(xi0)
                else:
                    xi0 = var_x[f"x({k_name_0})"]
                    xi1 = var_x[f"x({k_name_1})"]
                    cons.append(Not(xi1))
                    cons.append(xi0)
            else:
                constraints.append((current_c, v))

        if len(cons) > 0:
            constraints.append(current_c == z3.And(*cons))

    if len(all_c) > 1:
        constraints.append(z3.Or(all_c))
        for ci in all_c:
            other_c = set(all_c).difference({ci})
            constraints.append(z3.Implies(ci, z3.Not(z3.Or(other_c))))
    return all_c, constraints


def find_linf_tree_intersect(tree, sample, epsilon):
    leaves_0, min_corners_0, max_corners_0 = leaf_boxes(tree.tree_)

    maxi = []
    mini = []
    for i in range(sample.shape[1]):
        maxi.append(
            np.max([sample[0, i] * (1 - epsilon), sample[0, i] * (1 + epsilon)])
        )
        mini.append(
            np.min([sample[0, i] * (1 - epsilon), sample[0, i] * (1 + epsilon)])
        )

    mini = np.asarray(mini).reshape(1, -1)
    maxi = np.asarray(maxi).reshape(1, -1)

    intersect = box_intersection(min_corners_0, mini, max_corners_0, maxi)

    return intersect


def get_output(opt, c_weights):
    opt_vlues = {str(v): opt.model()[v] for v in opt.model() if "c" in str(v)}
    adv_weights = {
        ci: c_weights[ci]
        for ci, vi in opt_vlues.items()
        if vi and ci in c_weights.keys()
    }
    return adv_weights


def create_all_smt(clf, var_x, sample, epsilon, lower_bound = False):
    predict = clf.predict(sample)[0]
    print(predict)

    c_weights = {}
    all_c = []

    remove_c = []

    linf_conds = linf_const_x(var_x, sample, epsilon)
    all_thresh = get_ens_thresh(clf)
    x_conditions = create_x_conditions(var_x, all_thresh)

    core_constraints = set()
    soft_constraints = set()

    for ci in linf_conds:
        core_constraints.add(ci)
    for ci in x_conditions:
        core_constraints.add(ci)

    for i, est in enumerate(clf.estimators_):
        leaf = est.tree_.apply(sample.astype(np.float32))
        remove_c.append(f"c({i},{leaf[0]})")

        leaves = np.where(est.tree_.feature == -2)[0]
        intersect_i = find_linf_tree_intersect(est, sample, epsilon)

        for ci in leaves[np.reshape(intersect_i, -1)]:
            all_c.append(f"c({i},{ci})")

    remaining_c = set(all_c).difference(remove_c)
    # print("included c's: ", remaining_c)
    if len(remaining_c) == 0:
        print("No intersecting box exist.")
        return None, None, c_weights, all_c

    not_cs = []
    for i, estimator in enumerate(clf.estimators_):
        dict_boundaries = find_boundaries(estimator)
        _, constraints = create_smt_regions(dict_boundaries, var_x, i)

        list_c = [ci for ci in constraints if isinstance(ci, tuple)]
        for ci in list_c:
            c_weights[str(ci[0])] = ci[1]

        for ci in constraints:
            if isinstance(ci, tuple):
                if f"{ci[0]}" in all_c:
                    if f"{ci[0]}" not in remove_c:
                        if predict:
                            soft_constraints.add((ci[0], (1 - ci[1])))
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

    return list(core_constraints), list(soft_constraints), c_weights, all_c


def binerize_val(val, nbits):
    max_val = 2 ** (nbits) - 1

    val_ = int(val * max_val)
    val_b = [int(bi) for bi in "{0:b}".format(val_)]
    val_ = copy.deepcopy(val_b)
    while len(val_) < nbits:
        val_.insert(0, 0)
    return [bool(vi) for vi in val_]


def list_c_val(c_weights, nbits):
    list_c_ = []
    list_val_ = []
    for k, v in c_weights.items():
        list_val_.append(binerize_val(v, nbits))
        list_c_.append(Bool(k))
    return list_val_, list_c_


def sum_loop(xin, c_, n):
    x_ = copy.deepcopy(xin)
    for xi in x_:
        while len(xi) < n:
            xi.insert(0, False)

    constraints = set()
    d = {}
    c = {}
    x = {}
    xh = {}

    for seq_num in range(1, len(x_)):

        c[(seq_num, 0)] = Bool(f"csum_{seq_num}({0})")
        for i in range(1, n + 1):
            d[(seq_num, i)] = Bool(f"dsum_{seq_num}({i})")
            c[(seq_num, i)] = Bool(f"csum_{seq_num}({i})")
            x[(seq_num, i)] = Bool(f"xsum_{seq_num}({i})")
            xh[(seq_num, i)] = Bool(f"xhsum_{seq_num}({i})")

    for i in range(1, n + 1):
        d[(0, i)] = Bool(f"dsum_{0}({i})")
        xh[(0, i)] = Bool(f"xhsum_{0}({i})")

    for seq_num in range(1, len(x_)):

        constraints.add((Not(c[(seq_num, n)])))

        if seq_num == 1:
            for i in range(1, n + 1):
                cons = xh[(0, i)] if x_[0][i - 1] else Not(xh[(0, i)])
                constraints.add(cons)
                constraints.add(d[(0, i)] == And(c_[0], xh[(0, i)]))

        for i in range(1, n + 1):
            cons = xh[(seq_num, i)] if x_[seq_num][i - 1] else Not(xh[(seq_num, i)])
            constraints.add(cons)
            constraints.add(x[(seq_num, i)] == And(c_[seq_num], xh[(seq_num, i)]))

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
                    == (x[(seq_num, i)] == (d[(seq_num - 1, i)] == c[(seq_num, i)]))
                )
            )
    return constraints, len(x_) - 1


def get_value(model, seq_num, nbits, ntrees):
    dsum = [ci for ci in model if f"dsum_{seq_num}" in str(ci)]
    ldsum = list(zip(dsum, [model[ci] for ci in dsum]))
    output = 0

    for di in ldsum:
        ind = len(ldsum) - (int(str(di[0]).split("(")[1][:-1]))
        val = int(bool(di[1]))
        output += 2 ** (ind) * val

    max_val = 2 ** (nbits) - 1
    output = output / max_val / ntrees
    return output


def const_larger(nbits, ntrees, seq_num):

    new_nbits = int(np.ceil(np.log2(ntrees)) + nbits)
    vhalf = (2 ** (nbits) - 1) * ntrees / 2
    vb = [int(bi) for bi in "{0:b}".format(int(vhalf))]

    while len(vb) < new_nbits:
        vb.insert(0, 0)

    const = []

    vbset = []
    for i, vbi in enumerate(vb):
        vbset.append(Bool(f"vb({i + 1})"))
        if vbi:
            const.append(vbset[i])
        else:
            const.append(Not(vbset[i]))

    const_or = [Bool(f"dsum_{seq_num}(1)")]

    for i in range(1, new_nbits):
        const_and = []
        for j in range(i):
            const_and.append(vbset[j] == Bool(f"dsum_{seq_num}({j + 1})"))
        const_and.append(Bool(f"dsum_{seq_num}({i + 1})"))
        const_and.append(Not(vbset[i]))

        const_or.append(And(const_and))

    const.append(Bool("class") == Or(const_or))

    return const


def soft_attack(clf, sample, epsilon, var_x):
    core_constraints, soft_constraints, c_weights, all_c = create_all_smt(
        clf, var_x, sample, epsilon
    )
    s = Optimize()
    s.set("timeout", 5000)
    for ci in core_constraints:
        s.add(ci)
    for ci in soft_constraints:
        s.add_soft(ci[0], ci[1])

    print(s.check())
    return s, c_weights


def hard_attack(clf, sample, epsilon, var_x, nbits):
    core_constraints, soft_constraints, c_weights, all_c = create_all_smt(
        clf, var_x, sample, epsilon
    )

    ntrees = len(clf.estimators_)
    list_val_, list_c_ = list_c_val(c_weights, nbits)
    new_nbits = int(np.ceil(np.log2(ntrees)) + nbits)
    sum_const, seq_num = sum_loop(list_val_, list_c_, new_nbits)
    const_class = const_larger(nbits, ntrees, seq_num)

    s = Solver()

    s.set("timeout", 25000)
    for ci in core_constraints:
        s.add(ci)

    for ci in sum_const:
        s.add(ci)

    for ci in const_class:
        s.add(ci)

    s.add([Not(Bool("class")) if clf.predict(sample)[0] else Bool("class")])

    print(s.check())

    return s, c_weights, seq_num


def get_x_adv(solver, var_x, sample):
    x_adv = pd.DataFrame(({k[2:-1]: [solver.model()[v]] for k, v in var_x.items()}))
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
        col_i = np.asarray([ci for ci in columns if int(ci.split("_")[0]) == i])
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
