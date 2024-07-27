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
tree_h = 1
n_threads = 1





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


def VectMax2(bit_length, key,  *data):
    def reducer(x, y):
        b = x[0].__gt__(y[0], bit_length=bit_length)
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

    res = newton_div(temp_left, sfix(total_prefix_count_128)) + newton_div(temp_right, sfix(total_surfix_count_128))

    if single_thread:
        stop_timer(30)
    n = len(y)

    remove_bits = sfix.f
    # remove_bits = max(sfix.f + util.log2(n) - 31, 0)
    if remove_bits > 0:
        res = res.v.round(128, remove_bits)
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
    v = [g.if_else(aa, -1) for aa in a]
    v = Sort([g.bit_not()], *v, n_bits=[1])
    return v


def CropLayer(k, *v):
    if max_leaves:
        n = min(2 ** k, max_leaves)
    else:
        n = 2 ** k
    return [vv[:min(n, len(vv))] for vv in v]


#
# def TrainLeafNodes(h, g, y, NID):
#     assert len(g) == len(y)
#     assert len(g) == len(NID)
#     Label = GroupSum(g, y.bit_not()) < GroupSum(g, y)
#     return FormatLayer(h, g, NID, Label)

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

    def AttributeWiseTestSelection(self, g, x, y, node_size):
        if single_thread:
            start_timer(102)
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
        # s, t = GroupMax2(node_size, gg, s, t)
        if single_thread:
            stop_timer(3)
        if single_thread:
            stop_timer(102)
        return t, s

    def GlobalTestSelection(self, x, y, g, node_size):
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
                g, u[j], v[j], node_size)

        n = len(g)
        a, tt = [sint.Array(n) for i in range(2)]
        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m),
                              (t[j][:] for j in range(m)))
        if single_thread:
            stop_timer(103)
        return a[:], tt[:]

    def TrainInternalNodes(self, k, x, y, g, NID):
        if single_thread:
            start_timer(104)
        assert len(g) == len(y)
        for xx in x:
            assert len(xx) == len(g)
        node_size = MemValue(2 ** k)
        AID, Threshold = self.GlobalTestSelection(x, y, g, node_size)
        res = FormatLayer_without_crop(g[:], NID, AID, Threshold), AID, Threshold
        if single_thread:
            stop_timer(104)
        return res

    @method_block
    def train_layer(self, k):
        print_ln("training %s-th layer", k)
        if single_thread:
            start_timer(105)
        self.layer_matrix[k], AID, Threshold = \
            self.TrainInternalNodes(k, self.x, self.y, self.g, self.NID)
        if single_thread:
            start_timer(1)
        recover_AID = PermUtil.unapply(self.perms[0], AID).get_vector()
        recover_Threshold = PermUtil.unapply(self.perms[0], Threshold).get_vector()
        if single_thread:
            stop_timer(1)
        b = self.ApplyTests(self.x, recover_AID, recover_Threshold)
        if single_thread:
            start_timer(1)
        temp_b = PermUtil.apply(self.perms[0], b).get_vector()
        if single_thread:
            stop_timer(1)
        temp_b_not = temp_b.bit_not()
        self.g.assign(GroupFirstOne(self.g, temp_b_not) + GroupFirstOne(self.g, temp_b))
        self.NID.assign(2 ** k * temp_b + self.NID)
        perm = SortPerm(temp_b)
        self.g.assign(perm.apply(self.g))
        self.NID.assign(perm.apply(self.NID))
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
        self.g = sint.Array(n)
        self.g.assign_all(0)
        self.g[0] = 1
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
            self.train_layer(k)

        tree = self.get_tree(self.h)
        # test_poplar('train', tree, self.y, self.x,
        #             n_threads=self.n_threads)
        return tree

    def train_with_testing(self, *test_set):
        """ Train decision tree and test against test data.

        :param y: binary labels (list or sint vector)
        :param x: sample data (by attribute, list or
          :py:obj:`~Compiler.types.Matrix`)
        :returns: tree

        """
        for k in range(self.h):
            self.train_layer(k)
        tree = self.get_tree(self.h)
        output_tree(tree)
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
        if single_thread:
            start_timer(1)
        temp_y = PermUtil.apply(self.perms[0], self.y).get_vector()
        if single_thread:
            stop_timer(1)
        Layer[h] = TrainLeafNodes(h, self.g[:], temp_y, self.NID)
        return Layer


def TreeTraining(x, y, h, binary=False):
    return TreeTrainer(x, y, h, binary=binary).train()


def output_tree(layers):
    """ Print decision tree output by :py:class:`TreeTrainer`. """
    # print_ln('full model %s', util.reveal(layers))
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




# def test_poplar(name, layers, y, x, n_threads=None):
#     start_timer(100)
#     n = len(y)
#     x = x.transpose().reveal()
#     y = y.reveal()
#     guess = regint.Array(n)
#     truth = regint.Array(n)
#     correct = regint.Array(2)
#     parts = regint.Array(2)
#     layers = [Matrix.create_from(util.reveal(layer)) for layer in layers]
#     @for_range_multithread(n_threads, 1, n)
#     def _(i):
#         guess[i] = run_poplar([[part[:] for part in layer]
#                                for layer in layers], x[i]).reveal()
#         truth[i] = y[i].reveal()
#     @for_range(n)
#     def _(i):
#         parts[truth[i]] += 1
#         c = (guess[i].bit_xor(truth[i]).bit_not())
#         correct[truth[i]] += c
#     print_ln('%s for height %s: %s/%s (%s/%s, %s/%s)', name, len(layers) - 1,
#              sum(correct), n, correct[0], parts[0], correct[1], parts[1])
#     stop_timer(100)


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
        guess[i] = run_poplar([[part[:] for part in layer]
                               for layer in layers], x[i]).reveal()
        truth[i] = y[i].reveal()

    correct = 0
    for i in range(n):
        correct = correct + (guess[i] == truth[i])
    print_ln('%s for height %s: %s/%s', name, len(layers) - 1,
             sum(correct), n)
    stop_timer(100)
