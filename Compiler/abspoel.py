from queue import Queue

from Compiler.types import *
from Compiler.sorting import *
from Compiler.library import *
from Compiler import util, oram



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



def newton_div(x, y):
    n = 2 ** (sfix.f / 2)
    z = sfix(1/n, size=y.size)
    for i in range(util.log2(n) + 3):
        z = 2 * z - y * z * z
    return x * z

def PrefixSum(x):
    return x.get_vector().prefix_sum()



MIN_VALUE = -10000




class Node:
    def __init__(self):
        self.threshold = None
        self.attribute_id = None
        self.label = None
        self.left_node = None
        self.right_node = None
        self.h = None


class TreeTrainer:

    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h
        self.m = len(x)
        self.n = len(y)
        self.internel_node_number = 2 ** h - 1
        self.leaf_node_number = 2 ** h
        self.perms = Matrix(self.m, self.n, sint)
        self.gen_perm_for_attrbutes()
        self.thresholds_array = sint.Array(self.internel_node_number + self.leaf_node_number)
        self.aid_array = sint.Array(self.internel_node_number + self.leaf_node_number)
        self.label_array = sint.Array(self.internel_node_number + self.leaf_node_number)
        self.isbelong_array = sint.Matrix(self.internel_node_number + self.leaf_node_number, self.n)

        f = 2 * util.log2(self.n)
        sfix.set_precision(f)
        cfix.set_precision(f)


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
    def train_internal_node(self,  k):
        self.aid_array[k],  self.thresholds_array[k] = self.compute_attribute_and_threshold(self.x, self.y, self.isbelong_array[k])
        comparison_results = self.apply_tests(self.x, self.aid_array[k], self.thresholds_array[k])
        self.isbelong_array[2*k + 1] = self.isbelong_array[k] * (1 - comparison_results)
        self.isbelong_array[2*k + 2] = self.isbelong_array[k].get_vector() - self.isbelong_array[2*k + 1].get_vector()



    @method_block
    def train_leaf_node(self,  k):
        self.label_array[k] = self.compute_label(self.y, self.isbelong_array[k])



    def compute_attribute_and_threshold(self, x, y, is_belong):
        ginis = sint.Array(self.m)
        thresholds = sint.Array(self.m)
        is_belong = Array.create_from(is_belong)
        @for_range_multithread(n_threads, 1, self.m)
        def _(i):
            sorted_attr = PermUtil.apply(self.perms[i], x[i])
            sorted_y = PermUtil.apply(self.perms[i], y)
            sorted_is_belong = PermUtil.apply(self.perms[i], is_belong)
            ginis[i], thresholds[i] = self.compute_gini_and_threshold_for_attribute(sorted_attr, sorted_y, sorted_is_belong)
        aid, threshold = VectMax(ginis, [i for i in range(self.m)], thresholds)
        return MemValue(aid), MemValue(threshold)


    def compute_gini_and_threshold_for_attribute(self, attr, y, is_belong):
        total_count = sum(is_belong)
        total_prefix_count = PrefixSum(is_belong)
        total_surfix_count = (total_count - total_prefix_count)
        temp_left = sint(0, size=len(y))
        temp_right = sint(0, size=len(y))
        for i in range(label_number):
            y_i = (y.get_vector() == i) * is_belong
            label_count = sum(y_i)
            label_prefix_count = PrefixSum(y_i)
            label_surfix_count = label_count - label_prefix_count
            temp_left = label_prefix_count * label_prefix_count + temp_left
            temp_right = label_surfix_count * label_surfix_count + temp_right
        if single_thread:
            start_timer(30)
        ginis = newton_div(temp_left, sfix(total_prefix_count)) + newton_div(temp_right, sfix(total_surfix_count))
        if single_thread:
            stop_timer(30)
        thresholds = get_type(attr).Array(len(attr))
        thresholds[-1] = MIN_VALUE
        thresholds.assign_vector(attr.get_vector(size=len(attr) - 1) + \
                        attr.get_vector(size=len(attr) - 1, base=1))
        gini, threshold = VectMax(ginis, ginis, thresholds)
        return gini, threshold

    def compute_label(self, y, is_belong):
        count = sint.Array(label_number)
        for i in range(label_number):
            count[i] = sum((y.get_vector() == i) * is_belong)
        label = VectMax(count, [i for i in range(label_number)])
        return MemValue(label[0])

    def apply_tests(self, x, AID, Threshold):
        bits = AID == regint.inc(self.m, 0)
        bits = Matrix.create_from([bits])
        xx = bits.dot(x)
        res = 2 * xx[0] > Threshold
        return res


    def gen_perm_for_attrbutes(self):
        @for_range_multithread(n_threads, 1, self.m)
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
            return self.predict_with_node(x, node.left_node, is_belong_left) \
                   + self.predict_with_node(x, node.right_node, is_belong_right)
        else:
            return node.label * is_belong


    def test(self, name, y, x):
        n = len(y)
        x = x.transpose()
        truth = y.reveal()
        guess = regint.Array(n)
        @for_range_multithread(n_threads, 1, n)
        def _(i):
            guess[i] = self.predict(x[i]).reveal()

        correct = 0
        for i in range(n):
            correct = correct + (guess[i] == truth[i])
        print_ln('%s for height %s: %s/%s', name, self.h,
                 sum(correct), n)

    def output_tree(self):
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            if node.left_node is not None:
                print_ln("h == %s, aid == %s, threshold == %s", node.h, node.attribute_id.reveal(), node.threshold.reveal())
                queue.put(node.left_node)
                queue.put(node.right_node)
            else:
                print_ln("h == %s, label == %s", node.h, node.label.reveal())
