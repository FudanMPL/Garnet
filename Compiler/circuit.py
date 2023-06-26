"""
This module contains functionality using circuits in the so-called
`Bristol Fashion`_ format. You can download a few examples including
the ones used below into ``Programs/Circuits`` as follows::

    make Programs/Circuits

.. _`Bristol Fashion`: https://homes.esat.kuleuven.be/~nsmart/MPC

"""

from Compiler.GC.types import *
from Compiler.library import function_block
from Compiler import util
import itertools
import struct

class Circuit:
    """
    Use a Bristol Fashion circuit in a high-level program. The
    following example adds signed 64-bit inputs from two different
    parties and prints the result::

        from circuit import Circuit
        sb64 = sbits.get_type(64)
        adder = Circuit('adder64')
        a, b = [sbitvec(sb64.get_input_from(i)) for i in (0, 1)]
        print_ln('%s', adder(a, b).elements()[0].reveal())

    Circuits can also be executed in parallel as the following example
    shows::

        from circuit import Circuit
        sb128 = sbits.get_type(128)
        key = sb128(0x2b7e151628aed2a6abf7158809cf4f3c)
        plaintext = sb128(0x6bc1bee22e409f96e93d7e117393172a)
        n = 1000
        aes128 = Circuit('aes_128')
        ciphertexts = aes128(sbitvec([key] * n), sbitvec([plaintext] * n))
        ciphertexts.elements()[n - 1].reveal().print_reg()

    This executes AES-128 1000 times in parallel and then outputs the
    last result, which should be ``0x3ad77bb40d7a3660a89ecaf32466ef97``,
    one of the test vectors for AES-128.

    """

    def __init__(self, name):
        self.filename = 'Programs/Circuits/%s.txt' % name
        f = open(self.filename)
        self.functions = {}

    def __call__(self, *inputs):
        return self.run(*inputs)

    def run(self, *inputs):
        n = inputs[0][0].n
        if n not in self.functions:
            self.functions[n] = function_block(lambda *args:
                                               self.compile(*args))
        flat_res = self.functions[n](*itertools.chain(*inputs))
        res = []
        i = 0
        for l in self.n_output_wires:
            v = []
            for i in range(l):
                v.append(flat_res[i])
                i += 1
            res.append(sbitvec.from_vec(v))
        return util.untuplify(res)

    def compile(self, *all_inputs):
        f = open(self.filename)
        lines = iter(f)
        next_line = lambda: next(lines).split()
        n_gates, n_wires = (int(x) for x in next_line())
        self.n_wires = n_wires
        input_line = [int(x) for x in next_line()]
        n_inputs = input_line[0]
        n_input_wires = input_line[1:]
        assert(n_inputs == len(n_input_wires))
        inputs = []
        s = 0
        for n in n_input_wires:
            inputs.append(all_inputs[s:s + n])
            s += n
        output_line = [int(x) for x in next_line()]
        n_outputs = output_line[0]
        self.n_output_wires = output_line[1:]
        assert(n_outputs == len(self.n_output_wires))
        next(lines)

        wires = [None] * n_wires
        self.wires = wires
        i_wire = 0
        for input, input_wires in zip(inputs, n_input_wires):
            assert(len(input) == input_wires)
            for i, reg in enumerate(input):
                wires[i_wire] = reg
                i_wire += 1

        for i in range(n_gates):
            line = next_line()
            t = line[-1]
            if t in ('XOR', 'AND'):
                assert line[0] == '2'
                assert line[1] == '1'
                assert len(line) == 6
                ins = [wires[int(line[2 + i])] for i in range(2)]
                if t == 'XOR':
                    wires[int(line[4])] = ins[0] ^ ins[1]
                else:
                    wires[int(line[4])] = ins[0] & ins[1]
            elif t == 'INV':
                assert line[0] == '1'
                assert line[1] == '1'
                assert len(line) == 5
                wires[int(line[3])] = ~wires[int(line[2])]

        return self.wires[-sum(self.n_output_wires):]

Keccak_f = None

def sha3_256(x):
    """
    This function implements SHA3-256 for inputs of up to 1080 bits::

        from circuit import sha3_256
        a = sbitvec.from_vec([])
        b = sbitvec(sint(0xcc), 8, 8)
        for x in a, b:
            sha3_256(x).elements()[0].reveal().print_reg()

    This should output the first two test vectors of SHA3-256 in
    byte-reversed order::

        0x4a43f8804b0ad882fa493be44dff80f562d661a05647c15166d71ebff8c6ffa7
        0xf0d7aa0ab2d92d580bb080e17cbb52627932ba37f085d3931270d31c39357067

    """

    global Keccak_f
    if Keccak_f is None:
        # only one instance
        Keccak_f = Circuit('Keccak_f')

    # whole bytes
    assert len(x.v) % 8 == 0
    # only one block
    r = 1088
    assert len(x.v) < 1088
    if x.v:
        n = x.v[0].n
    else:
        n = 1
    d = sbitvec([sbits.get_type(8)(0x06)] * n)
    sbn = sbits.get_type(n)
    padding = [sbn(0)] * (r - 8 - len(x.v))
    P_flat = x.v + d.v + padding
    assert len(P_flat) == r
    P_flat[-1] = ~P_flat[-1]
    w = 64
    P1 = [P_flat[i * w:(i + 1) * w] for i in range(r // w)]

    S = [[[sbn(0) for i in range(w)] for i in range(5)] for i in range(5)]
    for x in range(5):
        for y in range(5):
            if x + 5 * y < r // w:
                for i in range(w):
                    S[x][y][i] ^= P1[x + 5 * y][i]

    def flatten(S):
        res = [None] * 1600
        for y in range(5):
            for x in range(5):
                for i in range(w):
                    j = (5 * y + x) * w + i // 8 * 8 + 7 - i % 8
                    res[1600 - 1 - j] = S[x][y][i]
        return res

    def unflatten(S_flat):
        res = [[[None] * w for j in range(5)] for i in range(5)]
        for y in range(5):
            for x in range(5):
                for i in range(w):
                    j = (5 * x + y) * w + i // 8 * 8 + 7 - i % 8
                    res[x][y][i] = S_flat[1600 - 1 -j]
        return res

    S = unflatten(Keccak_f(flatten(S)))

    Z = []
    while len(Z) <= 256:
        for y in range(5):
            for x in range(5):
                if x + 5 * y < r // w:
                    Z += S[y][x]
        if len(Z) <= 256:
            S = unflatten(Keccak_f(flatten(S)))
    return sbitvec.from_vec(Z[:256])

class ieee_float:
    """
    This gives access IEEE754 floating-point operations using Bristol
    Fashion circuits. The following example computes the standard
    deviation of 10 integers input by each of party 0 and 1::

        from circuit import ieee_float

        values = []

        for i in range(2):
            for j in range(10):
                values.append(sbitint.get_type(64).get_input_from(i))

        fvalues = [ieee_float(x) for x in values]

        avg = sum(fvalues) / ieee_float(len(fvalues))
        var = sum(x * x for x in fvalues) / ieee_float(len(fvalues)) - avg * avg
        stddev = var.sqrt()

        print_ln('avg: %s', avg.reveal())
        print_ln('var: %s', var.reveal())
        print_ln('stddev: %s', stddev.reveal())
    """

    _circuits = {}
    is_clear = False

    @classmethod
    def circuit(cls, name):
        if name not in cls._circuits:
            cls._circuits[name] = Circuit('FP-' + name)
        return cls._circuits[name]

    def __init__(self, value):
        if isinstance(value, (sbitint, sbitintvec)):
            self.value = self.circuit('i2f')(sbitvec.conv(value))
        elif isinstance(value, sbitvec):
            self.value = value
        elif util.is_constant_float(value):
            self.value = sbitvec(sbits.get_type(64)(
                struct.unpack('Q', struct.pack('d', value))[0]))
        else:
            raise Exception('cannot convert type %s' % type(value))

    def __add__(self, other):
        return ieee_float(self.circuit('add')(self.value, other.value))

    def __radd__(self, other):
        if util.is_zero(other):
            return self
        else:
            return NotImplemented

    def __neg__(self):
        v = self.value.v[:]
        v[-1] = ~v[-1]
        return ieee_float(sbitvec.from_vec(v))

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        return ieee_float(self.circuit('mul')(self.value, other.value))

    def __truediv__(self, other):
        return ieee_float(self.circuit('div')(self.value, other.value))

    def __eq__(self, other):
        res = sbitvec.from_vec(self.circuit('eq')(self.value,
                                                  other.value).v[:1])
        if res.v[0].n == 1:
            return res.elements()[0]
        else:
            return res

    def sqrt(self):
        return ieee_float(self.circuit('sqrt')(self.value))

    def to_int(self):
        res = sbitintvec.from_vec(self.circuit('f2i')(self.value))
        if res.v[0].n == 1:
            return res.elements()[0]
        else:
            return res

    def reveal(self):
        assert self.value.v[0].n == 1
        m = self.value.v[:52]
        e = self.value.v[52:63]
        s = [self.value.v[63]]
        m, e, s = [sbitvec.from_vec(x).elements()[0].reveal()
                   for x in (m, e, s)]
        return cbitfloat(2 ** 52 + m, e - 2 ** 10 - 51,
                         cbit((m.to_regint() == 0) * (e.to_regint() == 0)), s,
                         (e.to_regint() == 2 ** 11 - 1))
