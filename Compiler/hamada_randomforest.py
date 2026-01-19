from Compiler.types import *
from Compiler.library import *
from Compiler import util

import math

import hamada

debug = False


def _select_features(m, max_features=None, tree_id=0):
    """Deterministic feature sub-sampling (per-tree) to avoid requiring PRGs."""
    if max_features is None:
        # Default: sample 50% of attributes
        max_features = (m + 1) // 2
    max_features = max(1, min(int(max_features), m))
    return [(tree_id * max_features + i) % m for i in range(max_features)]


def _slice_matrix_features(x, feature_ids):
    """x is a Matrix (m x n) in hamada.mpc usage; return Matrix(len(feature_ids) x n)."""
    x = Matrix.create_from(x)
    return Matrix.create_from([x[i] for i in feature_ids])


class RandomForest:
    """
    Random forest wrapper over `hamada.TreeTrainer`.

    Notes:
    - Uses per-tree feature sub-sampling (random subspace). Bootstrap sampling is
      omitted (can be added later).
    - Each tree is a standard hamada decision tree; final prediction is majority vote.
    """

    def __init__(self, x=None, y=None, h=None, tree_number=None,
                 n_threads=1, max_features=None):
        self.h = h
        self.tree_number = tree_number
        self.n_threads = n_threads
        self.max_features = max_features
        self.trees = []  # list of (feature_ids, layers)

        if x is not None:
            self.y = Array.create_from(y)
            self.x = Matrix.create_from(x)
            self.m = len(self.x)
            self.n = len(self.y)

    def fit(self):
        assert self.tree_number is not None
        self.trees = []
        m = len(self.x)
        for t in range(self.tree_number):
            feature_ids = _select_features(m, self.max_features, tree_id=t)
            x_sub = _slice_matrix_features(self.x, feature_ids)
            trainer = hamada.TreeTrainer(x_sub, self.y, self.h, attr_lengths=None,
                                         n_threads=self.n_threads)
            # Ring32 compile safety: limit fixed-point precision for the underlying
            # gini/newton operations inside hamada training.
            sfix.set_precision(8, 23)
            cfix.set_precision(8, 23)
            layers = trainer.train()
            self.trees.append((feature_ids, layers))
        return self

    def _predict_one(self, sample):
        votes = sint.Array(hamada.label_number)
        votes.assign_all(0)
        for feature_ids, layers in self.trees:
            sub = Array.create_from([sample[i] for i in feature_ids])
            pred = hamada.run_decision_tree(layers, sub)
            # pred is secret; update votes obliviously
            for lbl in range(hamada.label_number):
                votes[lbl] = votes[lbl] + (pred == lbl)
        # argmax over votes
        best = sint(0)
        best_votes = sint(-1)
        for lbl in range(hamada.label_number):
            better = votes[lbl] > best_votes
            best = better.if_else(lbl, best)
            best_votes = better.if_else(votes[lbl], best_votes)
        return best

    def predict(self, x):
        x = Matrix.create_from(x).transpose()
        n = len(x)
        res = sint.Array(n)

        # Avoid multithread tapes because they can capture objects from other tapes.
        @for_range(n)
        def _(i):
            res[i] = self._predict_one(x[i])

        return res

    def test(self, x, y, set_name="test"):
        print_ln("randomforest test for %s set", set_name)
        y_pred = self.predict(x)
        pred_res = y_pred.reveal()
        y_true = Array.create_from(y).reveal()
        n = len(y_true)
        right = 0
        for i in range(n):
            right = right + (pred_res[i] == y_true[i])
        print_ln("accuracy: %s/%s", right, n)
        return y_pred

