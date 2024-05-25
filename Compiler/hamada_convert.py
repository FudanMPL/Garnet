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
n_threads = 4
tree_h = 6
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


class SortPerm:
    def __init__(self, x):
        B = sint.Matrix(len(x), 2)
        B.set_column(0, 1 - x.get_vector())
        B.set_column(1, x.get_vector())
        self.perm = Array.create_from(dest_comp(B))
    def apply(self, x):
        res = Array.create_from(x)
        reveal_sort(self.perm, res, False)
        return res
    def unapply(self, x):
        res = Array.create_from(x)
        reveal_sort(self.perm, res, True)
        return res

def Sort(keys, *to_sort, n_bits=None, time=False):
    if single_thread:
        start_timer(1)
    for k in keys:
        assert len(k) == len(keys[0])
    n_bits = n_bits or [None] * len(keys)
    if single_thread:
        start_timer(2)
    bs = Matrix.create_from(
        sum([k.get_vector().bit_decompose(nb)
             for k, nb in reversed(list(zip(keys, n_bits)))], []))
    if single_thread:
        stop_timer(2)
    if single_thread:
        start_timer(3)
    res = Matrix.create_from(to_sort)
    res = res.transpose()
    if single_thread:
        stop_timer(3)

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
    z = sfix(1/n, size=y.size)
    for i in range(util.log2(n) + 3):
        z = 2 * z - y * z * z
    return x * z

def ModifiedGini(g, y, debug=False):
    if single_thread:
        start_timer(20)
    assert len(g) == len(y)
    y_bit = math.ceil(math.log2(label_number)) + 1
    ones = sint(1, size=len(y))
    total_count = GroupSum(g, ones)
    total_prefix_count = GroupPrefixSum(g, ones).get_vector()
    total_surfix_count = (total_count - total_prefix_count).get_vector()
    label_prefix_count = [None for i in range(label_number)]
    label_surfix_count = [None for i in range(label_number)]
    for i in range(label_number):
        y_i = y.get_vector().__eq__(ones * i, bit_length=y_bit)
        label_count = GroupSum(g, y_i)
        label_prefix_count[i] = GroupPrefixSum(g, y_i)
        label_surfix_count[i] = label_count - label_prefix_count[i]
        label_prefix_count[i] = label_prefix_count[i].get_vector()
        label_surfix_count[i] = label_surfix_count[i].get_vector()
    change_machine_domain(128)
    n = len(y)
    f = 2 * util.log2(n)
    sfix.set_precision(f, k=f + util.log2(n))
    cfix.set_precision(f, k=f + util.log2(n))
    temp_left = sint(0, size=len(y))
    temp_right = sint(0, size=len(y))
    for i in range(label_number):
        label_prefix_count_128 = label_prefix_count[i].change_domain_from_to(32, 128)
        label_surfix_count_128 = label_surfix_count[i].change_domain_from_to(32, 128)
        temp_left = label_prefix_count_128 * label_prefix_count_128 + temp_left
        temp_right = label_surfix_count_128 * label_surfix_count_128 + temp_right

    total_prefix_count_128 = total_prefix_count.change_domain_from_to(32, 128)
    total_surfix_count_128 = total_surfix_count.change_domain_from_to(32, 128)

    if single_thread:
        start_timer(30)

    res = newton_div(temp_left, sfix(total_prefix_count_128)) + newton_div(temp_right, sfix(total_surfix_count_128))

    if single_thread:
        stop_timer(30)
    n = len(y)

    remove_bits = max(sfix.f + util.log2(n) - 31, 0)
    if remove_bits > 0:
        res = res.v.round(sfix.f + util.log2(n), remove_bits)
    else:
        res = res.v

    change_machine_domain(32)
    sfix.set_precision(16, 31)
    cfix.set_precision(16, 31)
    res = res.change_domain_from_to(128, 32)

    if single_thread:
        stop_timer(20)
    return res



MIN_VALUE = -10000

def FormatLayer(h, g, *a):
    return CropLayer(h, *FormatLayer_without_crop(g, *a))

def FormatLayer_without_crop(g, *a):
    for x in a:
        assert len(x) == len(g)
    v = [g.if_else(aa, 0) for aa in a]
    v = Sort([g.bit_not()], *v, n_bits=[1])
    return v

def CropLayer(k, *v):
    if max_leaves:
        n = min(2 ** k, max_leaves)
    else:
        n = 2 ** k
    return [vv[:min(n, len(vv))] for vv in v]


def TrainLeafNodes(h, g, y, NID):
    if single_thread:
        start_timer(106)
    assert len(g) == len(y)
    assert len(g) == len(NID)
    Label = sint(0, len(g))
    y_bit = util.log2(label_number)
    ones = sint(1, size=len(y))
    max_count = sint(0, size=len(y))
    for i in range(label_number):
        y_i = y.get_vector().__eq__(ones * i, bit_length=y_bit)
        count = GroupSum(g, y_i)
        comp = max_count < count
        Label = comp * i + (1 - comp) * Label
        max_count = comp * count + (1 - comp) * max_count
    res = FormatLayer(h, g, NID, Label)
    if single_thread:
        stop_timer(106)
    return res



class TreeTrainer:
    """ Decision tree training by `Hamada et al.`_

    :param x: sample data (by attribute, list or
      :py:obj:`~Compiler.types.Matrix`)
    :param y: binary labels (list or sint vector)
    :param h: height (int)
    :param binary: binary attributes instead of continuous
    :param attr_lengths: attribute description for mixed data
      (list of 0/1 for continuous/binary)
    :param n_threads: number of threads (default: single thread)

    .. _`Hamada et al.`: https://arxiv.org/abs/2112.12906

    """
    def ApplyTests(self, x, AID, Threshold):
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

        return 2 * xx < Threshold

    def AttributeWiseTestSelection(self, g, x, y):
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
        s = p[:].if_else(MIN_VALUE, s)
        s, t = GroupMax(gg, s, t)
        return t, s

    def GlobalTestSelection(self, x, y, g):
        assert len(y) == len(g)
        for xx in x:
            assert(len(xx) == len(g))
        m = len(x)
        n = len(y)
        u, t = [get_type(x).Matrix(m, n) for i in range(2)]
        v = get_type(y).Matrix(m, n)
        s = sfix.Matrix(m, n)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            u[j][:], v[j][:] = Sort((PrefixSum(g), x[j]), x[j], y, n_bits=[util.log2(len(y)), None])
            t[j][:], s[j][:] = self.AttributeWiseTestSelection(
                g, u[j], v[j])
        n = len(g)
        a, tt = [sint.Array(n) for i in range(2)]
        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m),
                              (t[j][:] for j in range(m)))
        return a[:], tt[:]

    def TrainInternalNodes(self, x, y, g, NID):
        assert len(g) == len(y)
        for xx in x:
            assert len(xx) == len(g)
        AID, Threshold = self.GlobalTestSelection(x, y, g)
        b = self.ApplyTests(x, AID, Threshold)
        return FormatLayer_without_crop(g[:], NID, AID, Threshold), b

    @method_block
    def train_layer(self, k):
        print_ln("training %s-th layer", k)
        x = self.x
        y = self.y
        g = self.g
        NID = self.NID
        layer_matrix = self.layer_matrix
        self.layer_matrix[k], b = \
            self.TrainInternalNodes(x, y, g, NID)
        NID[:] = 2 ** k * b + NID
        b_not = b.bit_not()
        g[:] = GroupFirstOne(g, b_not) + GroupFirstOne(g, b)
        y[:], g[:], NID[:], *xx = Sort([b], y, g, NID, *x, n_bits=[1])
        for i, xxx in enumerate(xx):
            x[i] = xxx

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
        self.g = sint.Array(n)
        self.g.assign_all(0)
        self.g[0] = 1
        self.NID = sint.Array(n)
        self.NID.assign_all(1)
        self.y = Array.create_from(y)
        self.x = Matrix.create_from(x)
        self.layer_matrix = sint.Tensor([h, 3, n])
        self.n_threads = n_threads
        self.debug_selection = False
        self.debug_threading = False
        self.debug_gini = True
        f = 2 * util.log2(n)
        sfix.set_precision(f)
        cfix.set_precision(f)

    def train(self):
        """ Train and return decision tree. """
        h = len(self.layer_matrix)
        @for_range(h)
        def _(k):
            self.train_layer(k)
        return self.get_tree(h)


    def get_tree(self, h):
        Layer = [None] * (h + 1)
        for k in range(h):
            Layer[k] = CropLayer(k, *self.layer_matrix[k])
        Layer[h] = TrainLeafNodes(h, self.g[:], self.y[:], self.NID)
        return Layer


def DecisionTreeTraining(x, y, h, binary=False):
    return TreeTrainer(x, y, h, binary=binary).train()


def output_decision_tree(layers):
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
    index = 1
    for k, layer in enumerate(layers[:-1]):
        assert len(layer) == 3
        for x in layer:
            assert len(x) <= 2 ** k
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

def test_decision_tree(name, layers, y, x, n_threads=None):
    start_timer(100)
    n = len(y)
    x = x.transpose().reveal()
    y = y.reveal()
    guess = regint.Array(n)
    truth = regint.Array(n)
    correct = regint.Array(2)
    parts = regint.Array(2)
    layers = [Matrix.create_from(util.reveal(layer)) for layer in layers]
    @for_range_multithread(n_threads, 1, n)
    def _(i):
        guess[i] = run_decision_tree([[part[:] for part in layer]
                                      for layer in layers], x[i]).reveal()
        truth[i] = y[i].reveal()
    @for_range(n)
    def _(i):
        parts[truth[i]] += 1
        c = (guess[i].bit_xor(truth[i]).bit_not())
        correct[truth[i]] += c
    print_ln('%s for height %s: %s/%s (%s/%s, %s/%s)', name, len(layers) - 1,
             sum(correct), n, correct[0], parts[0], correct[1], parts[1])
    stop_timer(100)
