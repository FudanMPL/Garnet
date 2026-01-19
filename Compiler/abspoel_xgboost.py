from queue import Queue

from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler import util, oram

# Keep this file as an extension of `abspoel.py` without importing optimizations
# from `ents_xgboost.py` (no change_domain_from_to/change_machine_domain).

debug = False
max_leaves = None
learning_rate = 0.5
n_threads = 4
single_thread = False
label_number = 2


def get_type(x):
    if isinstance(x, (Array, SubMultiArray)):
        return x.value_type
    elif isinstance(x, (tuple, list)):
        x = x[0] + x[-1]
        if util.is_constant(x):
            return cint
        else:
            return type(x)
    else:
        return type(x)


def pick(bits, x):
    if len(bits) == 1:
        return bits[0] * x[0]
    else:
        try:
            return x[0].dot_product(bits, x)
        except:
            return sum(aa * bb for aa, bb in zip(bits, x))


def VectMax(key, *data):
    def reducer(x, y):
        b = x[0] > y[0]
        return [b.if_else(xx, yy) for xx, yy in zip(x, y)]
    if debug:
        key = list(key)
        data = [list(x) for x in data]
        print_ln('vect max key=%s data=%s', util.reveal(key), util.reveal(data))
    return util.tree_reduce(reducer, zip(key, *data))[1:]


def PrefixSum(x):
    return x.get_vector().prefix_sum()


MIN_VALUE = -10000


def newton_div(x, y):
    # Same idea as in abspoel.py: reciprocal approximation without FPDiv.
    n = 2 ** (sfix.f / 2)
    z = sfix(1 / n, size=y.size)
    for i in range(util.log2(n) + 3):
        z = 2 * z - y * z * z
    return x * z


class Loss:
    @staticmethod
    def gradient(y, y_pred):
        raise NotImplementedError

    @staticmethod
    def hessian(y, y_pred):
        raise NotImplementedError


class SquareLoss(Loss):
    @staticmethod
    def gradient(y, y_pred):
        # support Array or vector
        if hasattr(y, 'get_vector'):
            y = y.get_vector()
        if hasattr(y_pred, 'get_vector'):
            y_pred = y_pred.get_vector()
        return y - y_pred

    @staticmethod
    def hessian(y, y_pred):
        if hasattr(y, 'get_vector'):
            y = y.get_vector()
        n = len(y)
        res = get_type(y).Array(n)
        res.assign_all(1)
        return res


class Node:
    def __init__(self):
        self.threshold = None
        self.attribute_id = None
        self.label = None  # sfix leaf weight
        self.left_node = None
        self.right_node = None
        self.h = None


class XGBoostTreeTrainer:
    """
    Implement XGBoost on top of the original `abspoel.py` node-based trainer.
    Key change: use gain from gradients/hessians and leaf weight = G/(H+lamb).

    To compile in ring32, we set a conservative sfix precision (k,f).
    """

    def __init__(self, x, y, y_pred, h, lamb=0.1, n_threads=4,
                 sfix_f=8, sfix_k=23):
        self.x = x
        self.y = y
        self.y_pred = y_pred
        self.h = h
        self.lamb = lamb
        self.n_threads = n_threads

        self.m = len(x)
        self.n = len(y)

        self.internel_node_number = 2 ** h - 1
        self.leaf_node_number = 2 ** h

        self.perms = Matrix(self.m, self.n, sint)
        self.gen_perm_for_attrbutes()

        self.thresholds_array = sint.Array(self.internel_node_number + self.leaf_node_number)
        self.aid_array = sint.Array(self.internel_node_number + self.leaf_node_number)
        self.label_array = sfix.Array(self.internel_node_number + self.leaf_node_number)
        self.isbelong_array = sint.Matrix(self.internel_node_number + self.leaf_node_number, self.n)

        # IMPORTANT for -R 32:
        # - Avoid FPDiv (we use newton_div instead).
        # - Keep (k+f) <= 31 and k >= f+15 for safe range.
        sfix.set_precision(sfix_f, sfix_k)
        cfix.set_precision(sfix_f, sfix_k)

    def train(self):
        is_belong = sint(1, self.n)
        self.isbelong_array[0] = Array.create_from(is_belong)
        @for_range(self.internel_node_number)
        def _(i):
            self.train_internal_node(i)
        @for_range(self.internel_node_number, self.internel_node_number + self.leaf_node_number)
        def _(i):
            self.train_leaf_node(i)
        self.to_node_structure()
        return self

    def to_node_structure(self):
        self.nodes = []
        for i in range(self.internel_node_number + self.leaf_node_number):
            self.nodes.append(Node())
            self.nodes[0].h = 0
        for i in range(self.internel_node_number):
            self.nodes[i].left_node = self.nodes[2 * i + 1]
            self.nodes[i].right_node = self.nodes[2 * i + 2]
            self.nodes[i].left_node.h = self.nodes[i].h + 1
            self.nodes[i].right_node.h = self.nodes[i].h + 1
        for i in range(self.internel_node_number):
            self.nodes[i].attribute_id = MemValue(self.aid_array[i])
            self.nodes[i].threshold = MemValue(self.thresholds_array[i])
        for i in range(self.internel_node_number, self.internel_node_number + self.leaf_node_number):
            self.nodes[i].label = MemValue(self.label_array[i])
        self.root = self.nodes[0]

    @method_block
    def train_internal_node(self, k):
        self.aid_array[k], self.thresholds_array[k] = self.compute_attribute_and_threshold(
            self.x, self.y, self.y_pred, self.isbelong_array[k])
        comparison_results = self.apply_tests(self.x, self.aid_array[k], self.thresholds_array[k])
        self.isbelong_array[2 * k + 1] = self.isbelong_array[k] * (1 - comparison_results)
        self.isbelong_array[2 * k + 2] = self.isbelong_array[k].get_vector() - self.isbelong_array[2 * k + 1].get_vector()

    @method_block
    def train_leaf_node(self, k):
        self.label_array[k] = self.compute_leaf_weight(self.y, self.y_pred, self.isbelong_array[k])

    def compute_attribute_and_threshold(self, x, y, y_pred, is_belong):
        gains = sfix.Array(self.m)
        thresholds = sint.Array(self.m)
        is_belong = Array.create_from(is_belong)

        # Avoid multithread tapes here because storing sfix results into arrays
        # from another tape triggers "Register from other tape" errors.
        @for_range(self.m)
        def _(i):
            sorted_attr = PermUtil.apply(self.perms[i], x[i])
            sorted_y = PermUtil.apply(self.perms[i], y)
            sorted_y_pred = PermUtil.apply(self.perms[i], y_pred)
            sorted_is_belong = PermUtil.apply(self.perms[i], is_belong)
            gains[i], thresholds[i] = self.compute_gain_and_threshold_for_attribute(
                sorted_attr, sorted_y, sorted_y_pred, sorted_is_belong)

        aid, threshold = VectMax(gains, [i for i in range(self.m)], thresholds)
        return MemValue(aid), MemValue(threshold)

    def compute_gain_and_threshold_for_attribute(self, attr, y, y_pred, is_belong):
        gradients = SquareLoss.gradient(y, y_pred)
        hessians = SquareLoss.hessian(y, y_pred)

        # mask by membership
        g = gradients * is_belong
        h = hessians * is_belong

        G = sum(g)
        H = sum(h)
        G_l = PrefixSum(g)
        H_l = PrefixSum(h)
        G_r = G - G_l
        H_r = H - H_l

        lamb = sfix(self.lamb)
        gain = newton_div(G_l * G_l, H_l + lamb) + newton_div(G_r * G_r, H_r + lamb)

        thresholds = get_type(attr).Array(len(attr))
        thresholds[-1] = MIN_VALUE
        thresholds.assign_vector(attr.get_vector(size=len(attr) - 1) +
                                 attr.get_vector(size=len(attr) - 1, base=1))

        # mask invalid splits (equal adjacent or last)
        invalid = sint.Array(len(attr))
        invalid.assign_all(0)
        invalid.assign_vector(attr.get_vector(size=len(attr) - 1) ==
                              attr.get_vector(size=len(attr) - 1, base=1))
        invalid[-1] = 1

        inv = invalid.get_vector()
        min_gain = sfix(MIN_VALUE, size=len(gain))
        gain = inv.if_else(min_gain, gain.get_vector())

        min_thr = sint(MIN_VALUE, size=len(thresholds))
        thresholds = inv.if_else(min_thr, thresholds.get_vector())

        best_gain, best_threshold = VectMax(gain, gain, thresholds)
        return best_gain, best_threshold

    def compute_leaf_weight(self, y, y_pred, is_belong):
        is_belong = Array.create_from(is_belong)
        gradients = SquareLoss.gradient(y, y_pred) * is_belong
        hessians = SquareLoss.hessian(y, y_pred) * is_belong
        G = sum(gradients)
        H = sum(hessians)
        lamb = sfix(self.lamb)
        lr = sfix(learning_rate)
        w = newton_div(G, H + lamb) * lr
        return MemValue(w)

    def apply_tests(self, x, AID, Threshold):
        bits = AID == regint.inc(self.m, 0)
        bits = Matrix.create_from([bits])
        xx = bits.dot(x)
        res = 2 * xx[0] > Threshold
        return res

    def gen_perm_for_attrbutes(self):
        @for_range_multithread(self.n_threads, 1, self.m)
        def _(i):
            self.perms.assign_part_vector(gen_perm_by_radix_sort(self.x[i]).get_vector(), i)

    def predict(self, x):
        is_belong = sint(1)
        return self.predict_with_node(x, self.root, is_belong)

    def predict_with_node(self, x, node, is_belong):
        if node.left_node is not None:
            bits = node.attribute_id == regint.inc(self.m, 0)
            attr_value = pick(bits, x)
            comparison_result = 2 * attr_value > node.threshold
            is_belong_left = is_belong * (1 - comparison_result)
            is_belong_right = is_belong - is_belong_left
            return self.predict_with_node(x, node.left_node, is_belong_left) + \
                self.predict_with_node(x, node.right_node, is_belong_right)
        else:
            return node.label * is_belong


class XGBoost:
    def __init__(self, x=None, y=None, h=None, tree_number=None, learning_rate=0.5,
                 n_threads=4, lamb=0.1):
        globals()['learning_rate'] = learning_rate
        self.h = h
        self.tree_number = tree_number
        self.n_threads = n_threads
        self.lamb = lamb
        self.trees = []
        if x is not None:
            self.y = Array.create_from(y)
            self.x = Matrix.create_from(x)
            self.n = len(y)

    def fit(self):
        y_pred = sfix.Array(self.n)
        y_pred.assign_all(0)
        datas = self.x.transpose()
        update_pred = sfix.Array(self.n)
        for i in range(self.tree_number):
            print_ln("Training the %s-th tree", i)
            tree = XGBoostTreeTrainer(self.x, self.y, y_pred, self.h, lamb=self.lamb, n_threads=self.n_threads)
            tree.train()
            @for_range(self.n)
            def _(j):
                update_pred[j] = tree.predict(datas[j])
            y_pred = y_pred + update_pred
            self.trees.append(tree)
        return self

    def predict(self, x):
        datas = Matrix.create_from(x).transpose()
        n = len(datas)
        y_pred = sfix.Array(n)
        y_pred.assign_all(0)
        @for_range(n)
        def _(i):
            for tree in self.trees:
                y_pred[i] = y_pred[i] + tree.predict(datas[i])
        return y_pred


