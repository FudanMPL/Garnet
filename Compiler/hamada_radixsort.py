from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler import util, oram
from Compiler.group_ops import *

from itertools import accumulate
import math

debug = False
debug_split = False
debug_layers = False
max_leaves = None
label_number = 2
single_thread = False
tree_h = 6
n_threads = 4
is_malicious = False




def Sort(keys, *to_sort, n_bits=None, time=False):
    if single_thread:
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
    if single_thread:
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




def newton_div(x, y):
    n = 2 ** (sfix.f / 2)
    z = sfix(1 / n, size=y.size)
    for i in range(util.log2(n) + 3):
        z = 2 * z - y * z * z
    return x * z


def ModifiedGini(g, y, debug=False):
    change_machine_domain(128)
    change_machine_domain(128)
    label_number = 2
    assert len(g) == len(y)
    y_bit = math.ceil(math.log2(label_number)) + 1
    ones = sint(1, size=len(y))
    total_count = GroupSum(g, ones)
    total_prefix_count = GroupPrefixSum(g, ones)
    total_surfix_count = (total_count - total_prefix_count)
    temp_left = sint(0, size=len(y))
    temp_right = sint(0, size=len(y))
    n = len(y)
    f = 2 * util.log2(n)
    sfix.set_precision(f, k=f + util.log2(n))
    cfix.set_precision(f, k=f + util.log2(n))
    for i in range(label_number):
        y_i = y.get_vector().__eq__(ones * i, bit_length=y_bit)
        label_count = GroupSum(g, y_i)
        label_prefix_count = GroupPrefixSum(g, y_i)
        label_surfix_count = label_count - label_prefix_count
        temp_left = label_prefix_count * label_prefix_count + temp_left
        temp_right = label_surfix_count * label_surfix_count + temp_right
    res = newton_div(temp_left, sfix(total_prefix_count)) + newton_div(temp_right, sfix(total_surfix_count))

    return res.v


MIN_VALUE = -10000


def FormatLayer(h, g, *a):
    return CropLayer(h, *FormatLayer_without_crop(g, *a))


def FormatLayer_without_crop(g, *a):
    for x in a:
        assert len(x) == len(g)
    v = [g.if_else(aa, -1) for aa in a]
    v = Sort([g.bit_not()], *v, n_bits=[1])
    return v


def CropLayer(k, *v):
    if max_leaves:
        n = min(2 ** k, max_leaves)
    else:
        n = 2 ** k
    return [vv[:min(n, len(vv))] for vv in v]




class TreeTrainer:
    def ApplyTests(self, x, AID, Threshold):
        if single_thread:
            start_timer(101)
        m = len(x)
        n = len(AID)
        assert len(AID) == len(Threshold)
        for xx in x:
            assert len(xx) == len(AID)
        e = sint.Matrix(m, n)
        AID = Array.create_from(AID)

        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            e[j][:] = AID[:] == j

        xx = sum(x[j] * e[j] for j in range(m))
        res = 2 * xx > Threshold
        if single_thread:
            stop_timer(101)
        return res

    def AttributeWiseTestSelection(self, g, x, y):
        if single_thread:
            start_timer(102)
        assert len(g) == len(x)
        assert len(g) == len(y)
        s = ModifiedGini(g, y, debug=debug)
        xx = x
        t = get_type(x).Array(len(x))
        t[-1] = x[-1]
        t.assign_vector(xx.get_vector(size=len(x) - 1) + \
                        xx.get_vector(size=len(x) - 1, base=1))
        gg = g
        p = sint.Array(len(x))
        p[-1] = 1
        p.assign_vector(gg.get_vector(base=1, size=len(x) - 1).bit_or(
            xx.get_vector(size=len(x) - 1) == \
            xx.get_vector(size=len(x) - 1, base=1)))
        break_point()
        s = p[:].if_else(MIN_VALUE, s)

        if single_thread:
            start_timer(3)
        s, t = GroupMax(gg, s, t)
        if single_thread:
            stop_timer(3)
        if single_thread:
            stop_timer(102)
        return t, s

    def GlobalSplitSelection(self, x, y, g):
        if single_thread:
            start_timer(103)
        assert len(y) == len(g)
        for xx in x:
            assert (len(xx) == len(g))

        m = len(x)
        n = len(y)
        u, t = [get_type(x).Matrix(m, n) for i in range(2)]
        v = get_type(y).Matrix(m, n)
        s = sint.Matrix(m, n)

        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            if single_thread:
                start_timer(1)
            u[j][:] = PermUtil.apply(self.perms[j], x[j])
            v[j][:] = PermUtil.apply(self.perms[j], y)
            if single_thread:
                stop_timer(1)
            t[j][:], s[j][:] = self.AttributeWiseTestSelection(
                g, u[j], v[j])

        n = len(g)
        a, tt = [sint.Array(n) for i in range(2)]
        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m),
                              (t[j][:] for j in range(m)))
        if single_thread:
            stop_timer(103)
        return a[:], tt[:]



    @method_block
    def train_internal_layer(self, k):
        print_ln("training %s-th layer", k)
        if single_thread:
            start_timer(105)
        n = len(self.NID)
        sorted_NID = PermUtil.apply(self.perms[0], self.NID)
        g = sint.Array(n)
        g[0] = 1
        g.get_sub(1, n).assign(sorted_NID.get_vector(0, n-1)!=sorted_NID.get_vector(1, n-1))

        AID, Threshold = self.GlobalSplitSelection(self.x, self.y, g)
        self.layer_matrix[k] = FormatLayer_without_crop(g[:], sorted_NID, AID, Threshold)

        unsorted_AID = PermUtil.unapply(self.perms[0], AID).get_vector()
        unsorted_Threshold = PermUtil.unapply(self.perms[0], Threshold).get_vector()
        b = self.ApplyTests(self.x, unsorted_AID, unsorted_Threshold)
        self.NID.assign(2 ** k * b + self.NID)
        self.update_perm_for_attrbutes(b)
        if single_thread:
            stop_timer(105)

    def __init__(self, x, y, h, binary=False, attr_lengths=None,
                 n_threads=None):
        assert not (binary and attr_lengths)
        if binary:
            attr_lengths = [1] * len(x)
        else:
            attr_lengths = attr_lengths or ([0] * len(x))
        for l in attr_lengths:
            assert l in (0, 1)
        self.attr_lengths = Array.create_from(regint(attr_lengths))
        Array.check_indices = False
        Matrix.disable_index_checks()
        for xx in x:
            assert len(xx) == len(y)
        n = len(y)
        self.NID = sint.Array(n)
        self.NID.assign_all(0)
        self.y = Array.create_from(y)
        self.x = Matrix.create_from(x)
        self.layer_matrix = sint.Tensor([h, 3, n])
        self.n_threads = n_threads
        self.debug_selection = False
        self.debug_threading = False
        self.debug_gini = True
        self.m = len(x)
        self.n = len(y)
        self.h = h
        self.perms = Matrix(self.m, self.n, sint)
        self.layers = [None] * (h + 1)
        self.gen_perm_for_attrbutes()

    def update_perm_for_attrbutes(self, b):
        if single_thread:
            start_timer(1)
        b = Array.create_from(b)

        @for_range_multithread(self.n_threads, 1, self.m)
        def _(i):
            temp_b = PermUtil.apply(self.perms[i], b)
            temp_perm = SortPerm(temp_b)
            self.perms.assign_part_vector(PermUtil.compose(self.perms[i], temp_perm).get_vector(), i)
        if single_thread:
            stop_timer(1)

    def gen_perm_for_attrbutes(self):
        if single_thread:
            start_timer(1)
        @for_range_multithread(self.n_threads, 1, self.m)
        def _(i):
            self.perms.assign_part_vector(gen_perm_by_radix_sort(self.x[i]).get_vector(), i)
        if single_thread:
            stop_timer(1)

    def train(self):
        """ Train and return decision tree. """
        for k in range(self.h):
            self.train_internal_layer(k)
            self.layers[k] = CropLayer(k, *self.layer_matrix[k])

        self.train_leaf_layer(self.h)

        return self.layers


    def train_leaf_layer(self, h):
        if single_thread:
            start_timer(106)
        n = len(self.NID)
        sorted_NID = PermUtil.apply(self.perms[0], self.NID)
        sorted_y = PermUtil.apply(self.perms[0], self.y)
        g = sint.Array(n)
        g[0] = 1
        g.get_sub(1, n).assign(sorted_NID.get_vector(0, n-1)!=sorted_NID.get_vector(1, n-1))
        Label = sint(0, n)
        y_bit = util.log2(label_number)
        ones = sint(1, size=len(sorted_y))
        max_count = sint(0, size=len(sorted_y))
        for i in range(label_number):
            y_i = sorted_y.get_vector().__eq__(ones * i, bit_length=y_bit)
            count = GroupSum(g, y_i)
            comp = max_count < count
            Label = comp * i + (1 - comp) * Label
            max_count = comp * count + (1 - comp) * max_count
        self.layers[h] = FormatLayer(h, g.get_vector(), sorted_NID, Label)
        if single_thread:
            stop_timer(106)



def TreeTraining(x, y, h, binary=False):
    return TreeTrainer(x, y, h, binary=binary).train()


def output_tree(layers):
    """ Print decision tree output by :py:class:`TreeTrainer`. """
    print_ln('full model %s', util.reveal(layers))
    for i, layer in enumerate(layers[:-1]):
        print_ln('level %s:', i)
        for j, x in enumerate(('NID', 'AID', 'Thr')):
            print_ln(' %s: %s', x, util.reveal(layer[j]))
    print_ln('leaves:')
    for j, x in enumerate(('NID', 'result')):
        print_ln(' %s: %s', x, util.reveal(layers[-1][j]))


def pick(bits, x):
    if len(bits) == 1:
        return bits[0] * x[0]
    else:
        try:
            return x[0].dot_product(bits, x)
        except:
            return sum(aa * bb for aa, bb in zip(bits, x))


def run_decision_tree(layers, data):
    """ Run decision tree against sample data.

    :param layers: tree output by :py:class:`TreeTrainer`
    :param data: sample data (:py:class:`~Compiler.types.Array`)
    :returns: binary label

    """
    h = len(layers) - 1
    index = 0
    for k, layer in enumerate(layers[:-1]):
        assert len(layer) == 3
        for x in layer:
            assert len(x) <= 2 ** k
        bits = layer[0].equal(index)
        threshold = pick(bits, layer[2])
        key_index = pick(bits, layer[1])
        if key_index.is_clear:
            key = data[key_index]
        else:
            key = pick(
                oram.demux(key_index.bit_decompose(util.log2(len(data)))), data)
        child = 2 * key > threshold
        index += child * 2 ** k
    bits = layers[h][0].equal(index)
    return pick(bits, layers[h][1])




def test_decision_tree(name, layers, y, x, n_threads=None):
    start_timer(100)
    n = len(y)
    x = x.transpose().reveal()
    y = y.reveal()
    guess = regint.Array(n)
    truth = regint.Array(n)
    layers = [Matrix.create_from(util.reveal(layer)) for layer in layers]

    @for_range_multithread(n_threads, 1, n)
    def _(i):
        guess[i] = run_decision_tree([[part[:] for part in layer]
                                      for layer in layers], x[i]).reveal()
        truth[i] = y[i].reveal()

    correct = 0
    for i in range(n):
        correct = correct + (guess[i] == truth[i])
    print_ln('%s for height %s: %s/%s', name, len(layers) - 1,
             sum(correct), n)
    stop_timer(100)