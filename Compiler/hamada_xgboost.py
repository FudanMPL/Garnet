from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler import util, oram
from Compiler.group_ops import *

import math

# Minimal XGBoost-style boosting built on top of the original `hamada.py` tree
# training structure. No change_domain_from_to / change_machine_domain.

debug = False
debug_split = False
debug_layers = False
max_leaves = None
learning_rate = 0.5


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


def Sort(keys, *to_sort, n_bits=None, time=False):
    if time:
        start_timer(1)
    for k in keys:
        assert len(k) == len(keys[0])
    n_bits = n_bits or [None] * len(keys)
    bs = Matrix.create_from(
        sum([k.get_vector().bit_decompose(nb)
             for k, nb in reversed(list(zip(keys, n_bits)))], []))
    res = Matrix.create_from(to_sort)
    res = res.transpose()
    radix_sort_from_matrix(bs, res)
    if time:
        stop_timer(1)
    return res.transpose()


def VectMax(key, *data):
    def reducer(x, y):
        b = x[0] > y[0]
        return [b.if_else(xx, yy) for xx, yy in zip(x, y)]
    if debug:
        key = list(key)
        data = [list(x) for x in data]
        print_ln('vect max key=%s data=%s', util.reveal(key), util.reveal(data))
    return util.tree_reduce(reducer, zip(key, *data))[1:]


MIN_VALUE = -10000


def CropLayer(k, *v):
    if max_leaves:
        n = min(2 ** k, max_leaves)
    else:
        n = 2 ** k
    return [vv[:min(n, len(vv))] for vv in v]


def pick(bits, x):
    if len(bits) == 1:
        return bits[0] * x[0]
    else:
        try:
            return x[0].dot_product(bits, x)
        except:
            return sum(aa * bb for aa, bb in zip(bits, x))


def newton_div(x, y):
    # Same idea as in hamada.py: reciprocal approximation without FPDiv.
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


def TrainLeafNodes(h, g, y, y_pred, NID, lamb=0.1):
    assert len(g) == len(y)
    assert len(g) == len(NID)
    gradients = SquareLoss.gradient(y, y_pred)
    hessians = SquareLoss.hessian(y, y_pred)
    G = GroupSum(g, gradients)
    H = GroupSum(g, hessians)
    lamb = sfix(lamb)
    lr = sfix(learning_rate)
    weight = newton_div(G, H + lamb) * lr
    # output (NID, weight) in the same layer style (masked/sorted by g)
    layer = []
    layer.append(NID)
    layer.append(weight)
    layer = [g.if_else(aa, -1) for aa in layer]
    perm = SortPerm(g.bit_not())
    layer = [perm.apply(aa) for aa in layer]
    return CropLayer(h, layer[0], layer[1])


class XGBoost:
    def __init__(self, x=None, y=None, h=None, tree_number=None, learning_rate=0.5,
                 binary=False, attr_lengths=None, n_threads=1):
        globals()['learning_rate'] = learning_rate
        self.h = h
        self.tree_number = tree_number
        self.n_threads = n_threads
        self.trees = []
        if x is not None:
            self.y = Array.create_from(y)
            self.x = Matrix.create_from(x)
            self.m = len(x)
            self.n = len(y)

    def fit(self):
        y_pred = sfix.Array(self.n)
        y_pred.assign_all(0)
        datas = self.x.transpose()
        update_pred = sfix.Array(self.n)
        for i in range(self.tree_number):
            print_ln("Training the %s-th tree", i)
            tree = XGBoostTree(self.x, self.y, y_pred, self.h, n_threads=self.n_threads)
            tree.fit()
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


class XGBoostTree:
    """
    Tree training structure mirrors `hamada.py` (NID + group-based selection),
    but uses XGBoost gain for split selection and sfix weights in leaves.
    """

    def __init__(self, x, y, y_pred, h, lamb=0.1, n_threads=1, sfix_f=8, sfix_k=23):
        self.x = Matrix.create_from(x)
        self.y = Array.create_from(y)
        self.y_pred = Array.create_from(y_pred)
        self.h = h
        self.lamb = lamb
        self.n_threads = n_threads
        self.m = len(x)
        self.n = len(y)

        # IMPORTANT for -R 32: keep (k+f) <= 31 and k >= f+15 for safe range.
        sfix.set_precision(sfix_f, sfix_k)
        cfix.set_precision(sfix_f, sfix_k)

        self.g = sint.Array(self.n)
        self.g.assign_all(0)
        self.g[0] = 1
        self.NID = sint.Array(self.n)
        self.NID.assign_all(1)

        self.layer_matrix = sint.Tensor([h, 3, self.n])

    def ApplyTests(self, x, AID, Threshold):
        m = len(x)
        n = len(AID)
        assert len(AID) == len(Threshold)
        e = sint.Matrix(m, n)
        AID = Array.create_from(AID)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            e[j][:] = AID[:] == j
        xx = sum(x[j] * e[j] for j in range(m))
        return 2 * xx < Threshold

    def Gain(self, g, y, y_pred):
        gradients = SquareLoss.gradient(y, y_pred)
        hessians = SquareLoss.hessian(y, y_pred)
        G = GroupSum(g, gradients)
        H = GroupSum(g, hessians)
        G_l = GroupPrefixSum(g, gradients)
        H_l = GroupPrefixSum(g, hessians)
        G_r = G - G_l
        H_r = H - H_l
        lamb = sfix(self.lamb)
        return newton_div(G_l * G_l, H_l + lamb) + newton_div(G_r * G_r, H_r + lamb)

    def AttributeWiseTestSelection(self, g, x, y, y_pred):
        s = self.Gain(g, y, y_pred)
        t = get_type(x).Array(len(x))
        t[-1] = MIN_VALUE
        t.assign_vector(x.get_vector(size=len(x) - 1) +
                        x.get_vector(size=len(x) - 1, base=1))
        p = sint.Array(len(x))
        p[-1] = 1
        p.assign_vector(g.get_vector(base=1, size=len(x) - 1).bit_or(
            x.get_vector(size=len(x) - 1) == x.get_vector(size=len(x) - 1, base=1)))
        s = p[:].if_else(sfix(MIN_VALUE), s)
        s, t = GroupMax(g, s, t)
        return t, s

    def GlobalTestSelection(self, x, y, y_pred, g):
        m = len(x)
        n = len(y)
        u, t = [get_type(x).Matrix(m, n) for _ in range(2)]
        v = get_type(y).Matrix(m, n)
        w = get_type(y_pred).Matrix(m, n)
        s = sfix.Matrix(m, n)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            u[j][:], v[j][:], w[j][:] = Sort(
                (PrefixSum(g), x[j]), x[j], y, y_pred,
                n_bits=[util.log2(len(y)), None])
            t[j][:], s[j][:] = self.AttributeWiseTestSelection(g, u[j], v[j], w[j])
        a, tt = [sint.Array(n) for _ in range(2)]
        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m), (t[j][:] for j in range(m)))
        return a[:], tt[:]

    @method_block
    def train_layer(self, k):
        AID, Threshold = self.GlobalTestSelection(self.x, self.y, self.y_pred, self.g)
        b = self.ApplyTests(self.x, AID, Threshold)
        self.layer_matrix[k][0][:] = self.NID[:]
        self.layer_matrix[k][1][:] = AID[:]
        self.layer_matrix[k][2][:] = Threshold[:]
        self.NID[:] = 2 ** k * b + self.NID
        b_not = b.bit_not()
        self.g[:] = GroupFirstOne(self.g, b_not) + GroupFirstOne(self.g, b)
        self.y[:], self.y_pred[:], self.g[:], self.NID[:], *xx = Sort([b], self.y, self.y_pred, self.g, self.NID, *self.x, n_bits=[1])
        for i, xxx in enumerate(xx):
            self.x[i] = xxx

    def fit(self):
        @for_range(self.h)
        def _(k):
            self.train_layer(k)
        # build compact layers
        self.layers = [None] * (self.h + 1)
        for k in range(self.h):
            self.layers[k] = CropLayer(k, *self.layer_matrix[k])
        self.layers[self.h] = TrainLeafNodes(self.h, self.g[:], self.y[:], self.y_pred[:], self.NID, lamb=self.lamb)
        return self

    def predict(self, data):
        layers = self.layers
        h = len(layers) - 1
        index = 1
        for k, layer in enumerate(layers[:-1]):
            bits = layer[0].equal(index, k)
            threshold = pick(bits, layer[2])
            key_index = pick(bits, layer[1])
            if key_index.is_clear:
                key = data[key_index]
            else:
                key = pick(
                    oram.demux(key_index.bit_decompose(util.log2(len(data)))), data)
            child = 2 * key < threshold
            index += child * 2 ** k
        bits = layers[h][0].equal(index, h)
        return pick(bits, layers[h][1])

