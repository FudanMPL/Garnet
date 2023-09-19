from Compiler.group_ops import *



def pick(bits, x):
    if len(bits) == 1:
        return bits[0] * x[0]
    else:
        try:
            return x[0].dot_product(bits, x)
        except:
            return sum(aa * bb for aa, bb in zip(bits, x))



def fast_select(idx, key, value_list):
    l = len(key)
    if l <= 2 ** 6:
        index_vector = key.get_vector().equal(idx)
        res = []
        for value in value_list:
            val = get_type(value).dot_product(index_vector, value)
            res.append(val)
        return res
    tmp = math.log2(l)
    d1 = tmp / 2
    d2 = tmp - d1
    d1 = int(2 ** d1)
    d2 = int(2 ** d2)
    key_matrix = sint.Matrix(rows=d1, columns=d2)
    key_matrix.assign(key)
    compare_vector = key_matrix.get_column(0)
    compare_result = compare_vector.get_vector() <= idx
    # print_ln("key = %s", key.reveal())
    # print_ln("idx = %s", idx.reveal())
    # print_ln("compare_result = %s", compare_result.reveal())
    index_vector = sint.Array(d1)
    index_vector[d1 - 1] = compare_result[d1 - 1]
    index_vector.assign_vector(compare_result.get_vector(size=d1-1, base=0) - compare_result.get_vector(size=d1-1, base=1), base=0)
    compare_vector2 = key_matrix.transpose().dot(index_vector)
    index_vector2 = compare_vector2.get_vector().equal(idx)
    # print_ln("index_vector = %s", index_vector.reveal())
    # print_ln("compare_vector2 = %s", compare_vector2.reveal())
    # print_ln("index_vector2 = %s", index_vector2.reveal())
    # print_ln("compute index = %s", sint.dot_product(key_matrix.transpose().dot(index_vector), index_vector2).reveal())
    res = []
    for value in value_list:
        tmp_matrix = get_type(value).Matrix(d1, d2)
        tmp_matrix.assign(value)
        val = get_type(value).dot_product(tmp_matrix.transpose().dot(index_vector), index_vector2)
        res.append(val)
    return res


class XGBoostInference:
    def __init__(self, h, attribute_number,  n_threads, tree_number, test_sample_number, attribute_max_values=None):
        self.h = h
        self.attribute_number = attribute_number
        self.attribute_max_values = attribute_max_values
        self.n_threads = n_threads
        self.tree_number = tree_number
        self.test_sample_number = test_sample_number
        self.trees = []

    def input_from(self, pid):
        for i in range(self.tree_number):
            tree = TreeInference(h=self.h, attribute_number=self.attribute_number, attribute_max_values=self.attribute_max_values)
            tree.input_from(pid)
            self.trees.append(tree)

    def predict(self, x):
        datas = x.transpose()
        n = len(datas)
        y_pred = sfix.Array(n)
        # @for_range_multithread(self.n_threads, 1, n)
        # def _(i):
        for i in range(n):
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



class TreeInference:
    def __init__(self, h, attribute_number, attribute_max_values=None):
        self.attribute_max_values = attribute_max_values
        self.attribute_number = attribute_number
        self.h = h
        self.layers = []

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

    def predict(self, data):
        layers = self.layers
        h = len(layers) - 1
        index = 0
        for k, layer in enumerate(layers[:-1]):
            assert len(layer) == 3
            for x in layer:
                assert len(x) <= 2 ** k

            # bits = layer[0].get_vector().equal(index)
            # threshold = pick(bits, layer[2])
            # aid = pick(bits, layer[1])
            aid, threshold  = fast_select(index, layer[0], layer[1:])
            if aid.is_clear:
                attr_value = data[aid]
            else:
                # attr_value = fast_select(aid, None, data)
                attr_value = pick(
                    oram.demux(aid.bit_decompose(util.log2(len(data)))), data)
            comparison_result = 2 * attr_value > threshold
            index += comparison_result * 2 ** k
        # bits = layers[h][0].get_vector().equal(index)
        # res = pick(bits, layers[h][1])
        res = fast_select(index, layers[h][0], layers[h][1:])[0]
        return res