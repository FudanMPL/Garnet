import itertools
from Compiler import types, library, instructions

from Compiler.types import *
from Compiler.library import *
from Compiler import util, oram

def dest_comp(B):
    Bt = B.transpose()
    St_flat = Bt.get_vector().prefix_sum()
    Tt_flat = Bt.get_vector() * St_flat.get_vector()
    Tt = types.Matrix(*Bt.sizes, B.value_type)
    Tt.assign_vector(Tt_flat)
    return sum(Tt) - 1

def reveal_sort(k, D, reverse=False):
    assert len(k) == len(D)
    library.break_point()
    shuffle = types.sint.get_secure_shuffle(len(k))
    k_prime = k.get_vector().secure_permute(shuffle).reveal()
    idx = types.Array.create_from(k_prime)
    if reverse:
        D.assign_vector(D.get_slice_vector(idx))
        library.break_point()
        D.secure_permute(shuffle, reverse=True)
    else:
        D.secure_permute(shuffle)
        library.break_point()
        v = D.get_vector()
        D.assign_slice_vector(idx, v)
    library.break_point()
    instructions.delshuffle(shuffle)

def radix_sort(k, D, n_bits=None, signed=True):
    assert len(k) == len(D)
    bs = types.Matrix.create_from(k.get_vector().bit_decompose(n_bits))
    if signed and len(bs) > 1:
        bs[-1][:] = bs[-1][:].bit_not()
    radix_sort_from_matrix(bs, D)

def radix_sort_from_matrix(bs, D):
    n = len(D)
    for b in bs:
        assert(len(b) == n)
    B = types.sint.Matrix(n, 2)
    h = types.Array.create_from(types.sint(types.regint.inc(n)))
    @library.for_range(len(bs))
    def _(i):
        b = bs[i]
        B.set_column(0, 1 - b.get_vector())
        B.set_column(1, b.get_vector())
        c = types.Array.create_from(dest_comp(B))
        reveal_sort(c, h, reverse=False)
        @library.if_e(i < len(bs) - 1)
        def _():
            reveal_sort(h, bs[i + 1], reverse=True)
        @library.else_
        def _():
            reveal_sort(h, D, reverse=True)


class SortPerm(Array):
    def __init__(self, gen_from_x=None, length=0):
        if gen_from_x is not None:
            super().__init__(len(gen_from_x), sint)
            B = sint.Matrix(len(gen_from_x), 2)
            B.set_column(0, 1 - gen_from_x.get_vector())
            B.set_column(1, gen_from_x.get_vector())
            self.assign(Array.create_from(dest_comp(B)))
        else:
            super().__init__(length, sint)

    def apply(self, x):
        res = Array.create_from(x)
        reveal_sort(self, res, False)
        return res

    def unapply(self, x):
        res = Array.create_from(x)
        reveal_sort(self, res, True)
        return res

    def compose(self, x):
        temp = self.unapply(x)
        res = SortPerm(length=len(temp))
        res.assign(temp)
        return res



# class SortPerm:
#     def __init__(self, x=None):
#         if x is not None:
#             B = sint.Matrix(len(x), 2)
#             B.set_column(0, 1 - x.get_vector())
#             B.set_column(1, x.get_vector())
#             self.perm = Array.create_from(dest_comp(B))
#         else:
#             self.perm = None
#
#     def apply(self, x):
#         res = Array.create_from(x)
#         reveal_sort(self.perm, res, False)
#         return res
#
#     def unapply(self, x):
#         res = Array.create_from(x)
#         reveal_sort(self.perm, res, True)
#         return res
#
#     def compose(self, x):
#         temp = self.unapply(x.perm)
#         res = SortPerm()
#         res.perm = temp
#         return res
#
#     def reveal(self):
#         return self.perm.reveal()
#
#     def assign(self, x):
#         return self.perm.assign(x.perm)

class PermUtil:
    @staticmethod
    def apply(arr1, arr2):
        res = Array.create_from(arr2)
        reveal_sort(arr1, res, False)
        return res

    @staticmethod
    def unapply(arr1, arr2):
        res = Array.create_from(arr2)
        reveal_sort(arr1, res, True)
        return res

    @staticmethod
    def compose(arr1, arr2):
        return SortPerm.unapply(arr1, arr2)



def gen_perm_by_radix_sort(k, n_bits=None, signed=True):
    bs = types.Matrix.create_from(k.get_vector().bit_decompose(n_bits))
    if signed and len(bs) > 1:
        bs[-1][:] = bs[-1][:].bit_not()
    perm = SortPerm(bs[0])
    @library.for_range(1,len(bs))
    def _(i):
        nonlocal perm
        b = perm.apply(bs[i])
        temp = SortPerm(b)
        # 这里必须使用assign, 因为for range关键字无法使用=来赋值array
        perm.assign(perm.compose(temp))
    return perm