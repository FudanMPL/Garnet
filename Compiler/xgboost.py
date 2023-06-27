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
    if time:
        start_timer(11)
    radix_sort_from_matrix(bs, res)
    if time:
        stop_timer(11)
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


def TrainLeafNodes(h, g, y, y_pred, NID, lamb=0.1):
    assert len(g) == len(y)
    assert len(g) == len(NID)
    gradients = SquareLoss.gradient(y, y_pred)
    hessians = SquareLoss.hessian(y, y_pred)
    G = GroupSum(g, gradients)
    H = GroupSum(g, hessians)
    change_machine_domain(128)
    G_128 = G.change_domain_from_to(32, 128)
    H_128 = H.change_domain_from_to(32, 128)
    Label_128 = G_128 / (H_128 + lamb)
    # print_ln("Label_128 = %s", Label_128.reveal())
    Label_128 = Label_128 * learning_rate
    # print_ln("Label_128 * learning_rate = %s", Label_128.reveal())
    change_machine_domain(32)
    Label = Label_128.change_domain_from_to(128, 32)
    # print_ln("Label = %s", Label.reveal())
    layer = []
    layer.append(NID)
    layer.append(Label)
    layer = [g.if_else(aa, -1) for aa in layer]
    perm = SortPerm(g.bit_not())
    layer = [perm.apply(aa) for aa in layer]
    return CropLayer(h, layer[0], layer[1])


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


class XGBoost:
    def __init__(self, x=None, y=None, h=None, tree_number=None, learning_rate=0.5, binary=False, attr_lengths=None,
                 n_threads=None,  attribute_number=None, attribute_max_values=None, test_sample_number=None):
        if x is None:  # only inference
            self.h = h
            self.attribute_number = attribute_number
            self.attribute_max_values = attribute_max_values
            self.n_threads = n_threads
            self.tree_number = tree_number
            self.test_sample_number = test_sample_number
            self.trees = []
        else:  # training
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
            self.tree_number = tree_number
            self.y = Array.create_from(y)
            self.x = Matrix.create_from(x)
            self.n_threads = n_threads
            self.m = len(x)
            self.n = len(y)
            self.h = h
            self.perms = Matrix(self.m, self.n, sint)
            self.gen_perm_for_attrbutes()
            self.learning_rate = learning_rate
            self.trees = []

    def input_from(self, pid):
        # write meta data in file for using tree-inference.x
        f = open("Player-Data/xgboost-meta", 'w')
        f.write(str(self.tree_number) + "\n")
        f.write(str(self.h) + "\n")
        f.write(str(self.attribute_number) + "\n")
        f.write(str(self.test_sample_number) + "\n")
        f.write(" ".join(str(self.attribute_max_values[i]) for i in range(self.attribute_number)))
        f.close()
        for i in range(self.tree_number):
            tree = XGBoostTree(h=self.h, attribute_number=self.attribute_number, attribute_max_values=self.attribute_max_values)
            tree.input_from(pid)
            self.trees.append(tree)


    def fit(self):
        y_pred = sfix.Array(self.n)
        datas = self.x.transpose()
        update_pred = sfix.Array(self.n)
        for i in range(self.tree_number):
            print_ln("Training the %s-th tree", i)
            tree = XGBoostTree(self.x, self.y, y_pred, self.h, self.perms)
            tree.fit()
            @for_range_multithread(self.n_threads, 1, self.n)
            def _(j):
                update_pred[j] = tree.predict(datas[j])
            y_pred = y_pred + update_pred
            self.trees.append(tree)

        self.test(self.x, self.y, "train")

    def predict(self, x):
        datas = x.transpose()
        n = len(datas)
        y_pred = sfix.Array(n)
        @for_range_multithread(self.n_threads, 1, n)
        def _(i):
        # for i in range(n):
            for tree in self.trees:
                y_pred[i] = y_pred[i] + tree.predict(datas[i])
        return y_pred

    def test(self, x, y, set_name="test"):
        print_ln("test for %s set", set_name)
        y_pred = self.predict(x)
        pred_res = y_pred.get_vector().v.round(32, sfix.f, nearest=True).reveal()
        y_true = y.reveal()
        print_ln("true y = %s", y_true)
        print_ln("pred y = %s", pred_res)
        print_ln("pred y = %s (not round)", y_pred.reveal())
        n = len(y)
        right = 0
        for i in range(n):
            right = right + (y_true[i] == pred_res[i])
        print_ln("accuracy: %s/%s", right, n)


    def gen_perm_for_attrbutes(self):
        @for_range_multithread(self.n_threads, 1, self.m)
        def _(i):
            self.perms.assign_part_vector(gen_perm_by_radix_sort(self.x[i]).get_vector(), i)

    def reveal_to(self, pid):
        for i in range(self.tree_number):
            self.trees[i].reveal_to(pid)

    def reveal_and_print(self):
        for i in range(self.tree_number):
            self.trees[i].reveal_and_print()


class XGBoostTree:

    def __init__(self, x=None, y=None, y_pred=None, h=None, perms=None, lamb=0.1, binary=False, attr_lengths=None,
                 n_threads=None, attribute_number=None, attribute_max_values=None):
        if x is None:
            self.attribute_max_values = attribute_max_values
            self.attribute_number = attribute_number
            self.h = h
            self.layers = []
        else:
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
            self.y_pred = Array.create_from(y_pred)
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
            self.perms = Matrix.create_from(perms)
            self.lamb = lamb

    def reveal_to(self, pid):
        for i in range(self.h):
            layer = self.layers[i]
            node_id = layer[0]
            attr_id = layer[1]
            thresholds = layer[2]
            node_number = len(layer[0])
            for j in range(node_number):
                node_id_plaintext = node_id[j].reveal_to(pid)
                node_id_plaintext.binary_output()
                attr_id_plaintext = attr_id[j].reveal_to(pid)
                attr_id_plaintext.binary_output()
                thresholds_plaintext = thresholds[j].reveal_to(pid)
                thresholds_plaintext.binary_output()
        leaf_layer = self.layers[self.h]
        node_id = leaf_layer[0]
        labels = leaf_layer[1]
        node_number = len(leaf_layer[0])
        for j in range(node_number):
            node_id_plaintext = node_id[j].reveal_to(pid)
            node_id_plaintext.binary_output()
            labels_plaintext = labels[j].reveal_to(pid)
            labels_plaintext.binary_output()

    def reveal_and_print(self):
        for i in range(self.h):
            layer = self.layers[i]
            node_id = layer[0]
            attr_id = layer[1]
            thresholds = layer[2]
            node_number = len(layer[0])
            # print_ln("node number = %s ", node_number)
            for j in range(node_number):
                node_id_plaintext = node_id[j].reveal()
                attr_id_plaintext = attr_id[j].reveal()
                thresholds_plaintext = thresholds[j].reveal() // cint(2)
                print_ln("%s %s %s", node_id_plaintext, attr_id_plaintext, thresholds_plaintext)

        leaf_layer = self.layers[self.h]
        node_id = leaf_layer[0]
        labels = leaf_layer[1]
        node_number = len(leaf_layer[0])
        # print_ln("node number = %s ", node_number)
        for j in range(node_number):
            node_id_plaintext = node_id[j].reveal()
            labels_plaintext = labels[j].reveal()
            print_ln("%s %s", node_id_plaintext, labels_plaintext)


    def input_from(self, pid):
        for i in range(self.h):
            layer = []
            node_number = 2 ** i
            node_id = sint.Array(size=node_number)
            attr_id = sint.Array(size=node_number)
            thresholds = sint.Array(size=node_number)
            for j in range(node_number):
                sample = sint.get_input_from(pid, size=3)
                node_id[j] = sample[0]
                attr_id[j] = sample[1]
                thresholds[j] = sample[2] * 2
            layer.append(node_id)
            layer.append(attr_id)
            layer.append(thresholds)
            self.layers.append(layer)
        node_number = 2 ** self.h
        node_id = sint.Array(size=node_number)
        labels = sfix.Array(size=node_number)
        for j in range(node_number):
            sample_id = sint.get_input_from(pid)
            sample_label = sfix.get_input_from(pid)
            node_id[j] = sample_id
            labels[j] = sample_label
        leaf_layer = []
        leaf_layer.append(node_id)
        leaf_layer.append(labels)
        self.layers.append(leaf_layer)
        return self


    def update_perm_for_attrbutes(self, b):
        # @for_range_multithread(self.n_threads, 1, self.m)
        # def _(i):
        for i in range(self.m):
            temp_b = PermUtil.apply(self.perms[i], b)
            temp_perm = SortPerm(temp_b)
            self.perms.assign_part_vector(PermUtil.compose(self.perms[i], temp_perm).get_vector(), i)


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
        return 2 * xx > Threshold



    def AttributeWiseTestSelection(self, g, x, y, y_pred, time=False, debug=False):
        assert len(g) == len(x)
        assert len(g) == len(y)
        s = self.Gain(g, y, y_pred, debug=debug)
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
        s, t = GroupMax(gg, s, t)
        return t, s

    def GlobalTestSelection(self, x, y, y_pred, g):
        assert len(y) == len(g)
        for xx in x:
            assert(len(xx) == len(g))

        m = len(x)
        n = len(y)
        u, t = [get_type(x).Matrix(m, n) for i in range(2)]
        v = get_type(y).Matrix(m, n)
        w = get_type(y).Matrix(m, n)
        s = sfix.Matrix(m, n)
        @for_range_multithread(self.n_threads, 1, m)
        def _(j):
            single = not self.n_threads or self.n_threads == 1
            u[j][:] = PermUtil.apply(self.perms[j], x[j])
            v[j][:] = PermUtil.apply(self.perms[j], y)
            w[j][:] = PermUtil.apply(self.perms[j], y_pred)
            t[j][:], s[j][:] = self.AttributeWiseTestSelection(
                g, u[j], v[j], w[j], time=single, debug=self.debug_selection)
        n = len(g)
        a, tt = [sint.Array(n) for i in range(2)]

        # print_ln('gain ' + ' '.join(str(i) + ':%s' for i in range(m)),
        #          *(ss[0].reveal() for ss in s))
        # print_ln('gain 2 ' + ' '.join(str(i) + ':%s' for i in range(m)),
        #          *(ss[-1].reveal() for ss in s))

        a[:], tt[:] = VectMax((s[j][:] for j in range(m)), range(m),
                              (t[j][:] for j in range(m)))
        break_point()
        return a[:], tt[:]

    def TrainInternalNodes(self, k, x, y, y_pred, g, NID):
        assert len(g) == len(y)
        for xx in x:
            assert len(xx) == len(g)
        AID, Threshold = self.GlobalTestSelection(x, y, y_pred, g)
        return FormatLayer_without_crop(g[:], NID, AID, Threshold), AID, Threshold

    def train_layer(self, k):
        self.layer_matrix[k], AID, Threshold = \
            self.TrainInternalNodes(k, self.x, self.y, self.y_pred, self.g, self.NID)
        recover_AID = PermUtil.unapply(self.perms[0], AID).get_vector()
        recover_Threshold = PermUtil.unapply(self.perms[0], Threshold).get_vector()
        b = self.ApplyTests(self.x, recover_AID, recover_Threshold)
        temp_b = PermUtil.apply(self.perms[0], b).get_vector()
        temp_b_not = temp_b.bit_not()
        self.g = GroupFirstOne(self.g, temp_b_not) + GroupFirstOne(self.g, temp_b)
        self.NID = 2 ** k * temp_b + self.NID
        perm = SortPerm(temp_b)
        self.g = perm.apply(self.g)
        self.NID = perm.apply(self.NID)
        self.update_perm_for_attrbutes(b)

    def predict(self, data):
        layers = self.layers
        h = len(layers) - 1
        index = 0
        for k, layer in enumerate(layers[:-1]):
            assert len(layer) == 3
            for x in layer:
                assert len(x) <= 2 ** k
            bits = layer[0].get_vector().equal(index)
            threshold = pick(bits, layer[2])
            key_index = pick(bits, layer[1])
            if key_index.is_clear:
                key = data[key_index]
            else:
                key = pick(
                    oram.demux(key_index.bit_decompose(util.log2(len(data)))), data)
            child = 2 * key > threshold
            index += child * 2 ** k
        bits = layers[h][0].get_vector().equal(index)
        res = pick(bits, layers[h][1])
        return res

    def gen_perm_for_attrbutes(self):
        @for_range_multithread(self.n_threads, 1, self.m)
        def _(i):
            self.perms.assign_part_vector(gen_perm_by_radix_sort(self.x[i]).get_vector(), i)

    def fit(self):
        """ Train and return decision tree. """
        for k in range(self.h):
            print_ln("training %s-th layer", k)
            self.train_layer(k)
        self.layers = self.get_tree(self.h)


    def get_tree(self, h):
        Layers = [None] * (h + 1)
        for k in range(h):
            Layers[k] = CropLayer(k, *self.layer_matrix[k])
        temp_y = PermUtil.apply(self.perms[0], self.y).get_vector()
        temp_y_pred = PermUtil.apply(self.perms[0], self.y_pred).get_vector()
        leaf_layer = TrainLeafNodes(h, self.g[:], temp_y, temp_y_pred, self.NID)
        leaf_layer[0] = Array.create_from(leaf_layer[0])
        leaf_layer[1] = Array.create_from(leaf_layer[1])
        Layers = [Matrix.create_from(layer) for layer in Layers[0:-1]]
        Layers.append(leaf_layer)
        return Layers

    # The implementation of Gain should be changed if the loss function is changed
    def Gain(self, g, y, y_pred, debug=False):
        assert len(g) == len(y)
        gradients = SquareLoss.gradient(y, y_pred)
        hessians = SquareLoss.hessian(y, y_pred)
        G = GroupSum(g, gradients)
        H = GroupSum(g, hessians)
        G_l = GroupPrefixSum(g, gradients)
        H_l = GroupPrefixSum(g, hessians)
        G_r = G - G_l
        H_r = H - H_l
        change_machine_domain(128)
        G_l_128 = G_l.change_domain_from_to(32, 128)
        H_l_128 = H_l.change_domain_from_to(32, 128)
        G_r_128 = G_r.change_domain_from_to(32, 128)
        H_r_128 = H_r.change_domain_from_to(32, 128)
        G_l_128_square = G_l_128 * G_l_128
        G_r_128_square = G_r_128 * G_r_128
        res = G_l_128_square / (H_l_128 + self.lamb) + G_r_128_square / (H_r_128 + self.lamb)
        n = len(y)
        res = res * 2 ** (31 - sfix.f - math.ceil(math.log(n)))
        res = res.v.round(128, sfix.f)
        change_machine_domain(32)
        res = res.change_domain_from_to(128, 32)
        return res



def output_xgboost(layers):
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
        return y - y_pred

    @staticmethod
    def hessian(y, y_pred):
        n = len(y)
        res = get_type(y).Array(n)
        res.assign_all(1)
        return res