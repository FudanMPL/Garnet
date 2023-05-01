from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler import util, oram

from itertools import accumulate
import math

debug = False
debug_split = False
debug_layers = False
max_leaves = None
label_number = 2
single_thread = False


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


def PrefixSum(x):
    return x.get_vector().prefix_sum()


def PrefixSumR(x):
    tmp = get_type(x).Array(len(x))
    tmp.assign_vector(x)
    break_point()
    tmp[:] = tmp.get_reverse_vector().prefix_sum()
    break_point()
    return tmp.get_reverse_vector()


def PrefixSum_inv(x):
    tmp = get_type(x).Array(len(x) + 1)
    tmp.assign_vector(x, base=1)
    tmp[0] = 0
    return tmp.get_vector(size=len(x), base=1) - tmp.get_vector(size=len(x))


def PrefixSumR_inv(x):
    tmp = get_type(x).Array(len(x) + 1)
    tmp.assign_vector(x)
    tmp[-1] = 0
    return tmp.get_vector(size=len(x)) - tmp.get_vector(base=1, size=len(x))


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

    for k in keys:
        assert len(k) == len(keys[0])
    n_bits = n_bits or [None] * len(keys)
    bs = Matrix.create_from(
        sum([k.get_vector().bit_decompose(nb)
             for k, nb in reversed(list(zip(keys, n_bits)))], []))
    res = Matrix.create_from(to_sort)
    res = res.transpose()
    if single_thread:
        start_timer(1)
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


def GroupSum(g, x):
    assert len(g) == len(x)
    p = PrefixSumR(x) * g
    pi = SortPerm(g.get_vector().bit_not())
    p1 = pi.apply(p)
    s1 = PrefixSumR_inv(p1)
    d1 = PrefixSum_inv(s1)
    d = pi.unapply(d1) * g
    return PrefixSum(d)


def GroupPrefixSum(g, x):
    assert len(g) == len(x)
    s = get_type(x).Array(len(x) + 1)
    s[0] = 0
    s.assign_vector(PrefixSum(x), base=1)
    q = get_type(s).Array(len(x))
    q.assign_vector(s.get_vector(size=len(x)) * g)
    return s.get_vector(size=len(x), base=1) - GroupSum(g, q)


def GroupMax(g, keys, *x):
    if debug:
        print_ln('group max input g=%s keys=%s x=%s', util.reveal(g),
                 util.reveal(keys), util.reveal(x))
    assert len(keys) == len(g)
    for xx in x:
        assert len(xx) == len(g)
    n = len(g)
    m = int(math.ceil(math.log(n, 2)))
    keys = Array.create_from(keys)
    x = [Array.create_from(xx) for xx in x]
    g_new = Array.create_from(g)
    g_old = g_new.same_shape()
    for d in range(m):
        w = 2 ** d
        g_old[:] = g_new[:]
        break_point()
        vsize = n - w
        g_new.assign_vector(g_old.get_vector(size=vsize).bit_or(
            g_old.get_vector(size=vsize, base=w)), base=w)
        b = keys.get_vector(size=vsize) > keys.get_vector(size=vsize, base=w)
        for xx in [keys] + x:
            a = b.if_else(xx.get_vector(size=vsize),
                          xx.get_vector(size=vsize, base=w))
            xx.assign_vector(g_old.get_vector(size=vsize, base=w).if_else(
                xx.get_vector(size=vsize, base=w), a), base=w)
        break_point()
        if debug:
            print_ln('group max w=%s b=%s a=%s keys=%s x=%s g=%s', w, b.reveal(),
                     util.reveal(a), util.reveal(keys),
                     util.reveal(x), g_new.reveal())
    t = sint.Array(len(g))
    t[-1] = 1
    t.assign_vector(g.get_vector(size=n - 1, base=1))
    if debug:
        print_ln('group max end g=%s t=%s keys=%s x=%s', util.reveal(g),
                 util.reveal(t), util.reveal(keys), util.reveal(x))
    return [GroupSum(g, t[:] * xx) for xx in [keys] + x]


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
    sfix.set_precision(f)
    cfix.set_precision(f)
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
    # res = sfix(temp_left * total_surfix_count_128 + temp_right * total_prefix_count_128) / (total_prefix_count_128 * total_surfix_count_128)

    res = newton_div(temp_left, sfix(total_prefix_count_128)) + newton_div(temp_right, sfix(total_surfix_count_128))
    # print_ln("res1 = %s", res.reveal())
    if single_thread:
        stop_timer(30)
    n = len(y)

    remove_bits = max(sfix.f + util.log2(n) - 31, 0)
    if remove_bits > 0:
        res = res.v.round(128, remove_bits)
    else:
        res = res.v
    # print_ln("res2 = %s", res.reveal())
    change_machine_domain(32)
    sfix.set_precision(16, 31)
    cfix.set_precision(16, 31)
    res = res.change_domain_from_to(128, 32)
    # print_ln("res2 = %s", res.reveal())
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
    assert len(g) == len(y)
    assert len(g) == len(NID)
    Label = sint(0, len(g))
    y_bit = math.ceil(math.log2(label_number))
    ones = sint(1, size=len(y))
    max_count = sint(0, size=len(y))
    for i in range(label_number):
        y_i = y.get_vector().__eq__(ones * i, bit_length=y_bit)
        count = GroupSum(g, y_i)
        comp = max_count < count
        Label = comp * i + (1 - comp) * Label
        max_count = comp * count + (1 - comp) * max_count
    return FormatLayer(h, g, NID, Label)


def GroupSame(g, y):
    assert len(g) == len(y)
    s = GroupSum(g, [sint(1)] * len(g))
    s0 = GroupSum(g, y.bit_not())
    s1 = GroupSum(g, y)
    if debug_split:
        print_ln('group same g=%s', util.reveal(g))
        print_ln('group same y=%s', util.reveal(y))
    return (s == s0).bit_or(s == s1)


def GroupFirstOne(g, b):
    assert len(g) == len(b)
    s = GroupPrefixSum(g, b)
    return s * b == 1

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
        if debug:
            print_ln('apply e=%s xx=%s', util.reveal(e), util.reveal(xx))
        return 2 * xx < Threshold

    def AttributeWiseTestSelection(self, g, x, y, time=False, debug=False):
        assert len(g) == len(x)
        assert len(g) == len(y)

        s = ModifiedGini(g, y, debug=debug)
        xx = x
        t = get_type(x).Array(len(x))
        t[-1] = MIN_VALUE
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
        t = p[:].if_else(MIN_VALUE, t[:])
        if single_thread:
            start_timer(3)
        s, t = GroupMax(gg, s, t)
        if single_thread:
            stop_timer(3)

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
        gid = PrefixSum(g)
        gid = Array.create_from(gid)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            single = not self.n_threads or self.n_threads == 1

            u[j][:], v[j][:] = Sort((gid, x[j]), x[j], y, n_bits=[util.log2(n), None], time=single)

            t[j][:], s[j][:] = self.AttributeWiseTestSelection(
                g, u[j], v[j], time=single, debug=self.debug_selection)

        n = len(g)
        a, tt = [sint.Array(n) for i in range(2)]
        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m),
                              (t[j][:] for j in range(m)))
        return a[:], tt[:]

    def TrainInternalNodes(self, k, x, y, g, NID):
        assert len(g) == len(y)
        for xx in x:
            assert len(xx) == len(g)
        AID, Threshold = self.GlobalTestSelection(x, y, g)
        s = GroupSame(g[:], y[:])
        AID, Threshold = s.if_else(0, AID), s.if_else(MIN_VALUE, Threshold)
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
            self.TrainInternalNodes(k, x, y, g, NID)
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


    def train(self):
        """ Train and return decision tree. """
        h = len(self.layer_matrix)
        for k in range(h):
            self.train_layer(k)
        return self.get_tree(h)

    def train_with_testing(self, *test_set):
        """ Train decision tree and test against test data.

        :param y: binary labels (list or sint vector)
        :param x: sample data (by attribute, list or
          :py:obj:`~Compiler.types.Matrix`)
        :returns: tree

        """
        for k in range(len(self.layer_matrix)):
            self.train_layer(k)
            tree = self.get_tree(k + 1)
            output_decision_tree(tree)
            test_decision_tree('train', tree, self.y, self.x,
                               n_threads=self.n_threads)
            if test_set:
                test_decision_tree('test', tree, *test_set,
                                   n_threads=self.n_threads)
        return tree

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
