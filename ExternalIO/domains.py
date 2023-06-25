import struct

class Domain:
    def __init__(self, value=0):
        self.v = int(value % self.modulus)
        assert(self.v >= 0)

    def __add__(self, other):
        try:
            res = self.v + other.v
        except:
            res = self.v + other
        return type(self)(res)

    def __mul__(self, other):
        try:
            res = self.v * other.v
        except:
            res = self.v * other
        return type(self)(res)

    __radd__ = __add__

    def __eq__(self, other):
        return self.v == other.v

    def __neq__(self, other):
        return self.v != other.v

    def unpack(self, os):
        self.v = 0
        buf = os.consume(self.n_bytes)
        for i, b in enumerate(buf):
            self.v += b << (i * 8)

    def pack(self, os):
        v = self.v
        temp_buf = []
        for i in range(self.n_bytes):
            temp_buf.append(v & 0xff)
            v >>= 8
        #Instead of using python a loop per value we let struct pack handle all it
        os.buf += struct.pack('<{}B'.format(len(temp_buf)), *tuple(temp_buf))

def Z2(k):
    class Z(Domain):
        modulus = 2 ** k
        n_words = (k + 63) // 64
        n_bytes = (k + 7) // 8

    return Z

def Fp(mod):
    import gmpy2

    class Fp(Domain):
        modulus = mod
        n_words = (modulus.bit_length() + 63) // 64
        n_bytes = 8 * n_words
        R = 2 ** (64 * n_words) % modulus
        R_inv = gmpy2.invert(R, modulus)

        def unpack(self, os):
            Domain.unpack(self, os)
            self.v = self.v * self.R_inv % self.modulus

        def pack(self, os):
            Domain.pack(type(self)(self.v * self.R), os)

    return Fp
