"""
This module defines functions directly available in high-level programs,
in particularly providing flow control and output.
"""

from Compiler.types import cint,sint,cfix,sfix,sfloat,MPCThread,Array,MemValue,cgf2n,sgf2n,_number,_mem,_register,regint,Matrix,_types, cfloat, _single, localint, personal, copy_doc, _vec
from Compiler.instructions import *
from Compiler.util import tuplify,untuplify,is_zero
from Compiler.allocator import RegintOptimizer
from Compiler import instructions,instructions_base,comparison,program,util
import inspect,math
import random
import collections
import operator
import copy
from functools import reduce





def change_machine_domain(k):
    break_point()
    get_program().bit_length = k - 1
    get_program().options.ring = k
    cmd(k)

def get_program():
    return instructions.program
def get_tape():
    return get_program().curr_tape
def get_block():
    return get_program().curr_block

def vectorize(function):
    def vectorized_function(*args, **kwargs):
        if len(args) > 0 and 'size' in dir(args[0]):
            instructions_base.set_global_vector_size(args[0].size)
            res = function(*args, **kwargs)
            instructions_base.reset_global_vector_size()
        elif 'size' in kwargs:
            instructions_base.set_global_vector_size(kwargs['size'])
            del kwargs['size']
            res = function(*args, **kwargs)
            instructions_base.reset_global_vector_size()
        else:
            res = function(*args, **kwargs)
        return res
    vectorized_function.__name__ = function.__name__
    copy_doc(vectorized_function, function)
    return vectorized_function

def set_instruction_type(function):
    def instruction_typed_function(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], program.Tape.Register):
            if args[0].is_gf2n:
                instructions_base.set_global_instruction_type('gf2n')
            else:
                instructions_base.set_global_instruction_type('modp')                
            res = function(*args, **kwargs)
            instructions_base.reset_global_instruction_type()
        else:
            res = function(*args, **kwargs)
        return res
    instruction_typed_function.__name__ = function.__name__
    return instruction_typed_function


def _expand_to_print(val):
    return ('[' + ', '.join('%s' for i in range(len(val))) + ']',) + tuple(val)

def print_str(s, *args):
    """ Print a string, with optional args for adding
    variables/registers with ``%s``. """
    def print_plain_str(ss):
        """ Print a plain string (no custom formatting options) """
        ss = bytearray(ss, 'utf8')
        i = 1
        while 4*i <= len(ss):
            print_char4(ss[4*(i-1):4*i])
            i += 1
        i = 4*(i-1)
        while i < len(ss):
            print_char(ss[i])
            i += 1

    if len(args) != s.count('%s'):
        raise CompilerError('Incorrect number of arguments for string format:', s)
    substrings = s.split('%s')
    for i,ss in enumerate(substrings):
        print_plain_str(ss)
        if i < len(args):
            if isinstance(args[i], MemValue):
                val = args[i].read()
            else:
                val = args[i]
            if isinstance(val, program.Tape.Register):
                if val.is_clear:
                    val.print_reg_plain()
                else:
                    raise CompilerError('Cannot print secret value:', args[i])
            elif isinstance(val, cfix):
                val.print_plain()
            elif isinstance(val, sfix) or isinstance(val, sfloat):
                raise CompilerError('Cannot print secret value:', args[i])
            elif isinstance(val, cfloat):
                val.print_float_plain()
            elif isinstance(val, (list, tuple, Array)):
                print_str(*_expand_to_print(val))
            else:
                try:
                    val.output()
                except AttributeError:
                    print_plain_str(str(val))

def print_ln(s='', *args):
    """ Print line, with optional args for adding variables/registers
    with ``%s``. By default only player 0 outputs, but the ``-I``
    command-line option changes that.

    :param s: Python string with same number of ``%s`` as length of :py:obj:`args`
    :param args: list of public values (regint/cint/int/cfix/cfloat/localint)

    Example:

    .. code::

        print_ln('a is %s.', a.reveal())
    """
    print_str(s + '\n', *args)

def print_ln_if(cond, ss, *args):
    """ Print line if :py:obj:`cond` is true. The further arguments
    are treated as in :py:func:`print_str`/:py:func:`print_ln`.

    :param cond: regint/cint/int/localint
    :param ss: Python string
    :param args: list of public values

    Example:

    .. code::

        print_ln_if(get_player_id() == 0, 'Player 0 here')
    """
    print_str_if(cond, ss + '\n', *args)

def print_str_if(cond, ss, *args):
    """ Print string conditionally. See :py:func:`print_ln_if` for details. """
    if util.is_constant(cond):
        if cond:
            print_str(ss, *args)
    else:
        subs = ss.split('%s')
        assert len(subs) == len(args) + 1
        if isinstance(cond, localint):
            cond = cond._v
        for i, s in enumerate(subs):
            if i != 0:
                val = args[i - 1]
                try:
                    val.output_if(cond)
                except:
                    if isinstance(val, (list, tuple, Array)):
                        print_str_if(cond, *_expand_to_print(val))
                    else:
                        print_str_if(cond, str(val))
            s = bytearray(s, 'utf8')
            s += b'\0' * ((-len(s)) % 4)
            while s:
                cond.print_if(s[:4])
                s = s[4:]

def print_ln_to(player, ss, *args):
    """ Print line at :py:obj:`player` only. Note that printing is
    disabled by default except at player 0. Activate interactive mode
    with `-I` to enable it for all players.

    :param player: int
    :param ss: Python string
    :param args: list of values known to :py:obj:`player`

    Example::

        print_ln_to(player, 'output for %s: %s', player, x.reveal_to(player))
    """
    cond = player == get_player_id()
    new_args = []
    for arg in args:
        if isinstance(arg, personal):
            if util.is_constant(arg.player) ^ util.is_constant(player):
                match = False
            else:
                if util.is_constant(player):
                    match = arg.player == player
                else:
                    match = id(arg.player) == id(player)
            if not match:
                raise CompilerError('player mismatch in personal printing')
            new_args.append(arg._v)
        else:
            new_args.append(arg)
    print_ln_if(cond, ss, *new_args)

def print_float_precision(n):
    """ Set the precision for floating-point printing.

    :param n: number of digits (int) """
    print_float_prec(n)

def runtime_error(msg='', *args):
    """ Print an error message and abort the runtime.
    Parameters work as in :py:func:`print_ln` """
    print_str('User exception: ')
    print_ln(msg, *args)
    crash()

def runtime_error_if(condition, msg='', *args):
    """ Conditionally print an error message and abort the runtime.

    :param condition: regint/cint/int/cbit
    :param msg: message
    :param args: list of public values to fit ``%s`` in the message

    """
    print_ln_if(condition, msg, *args)
    crash(condition)

def crash(condition=None):
    """ Crash virtual machine.

    :param condition: crash if true (default: true)

    """
    if isinstance(condition, localint):
        # allow crash on local values
        condition = condition._v
    if condition is None:
        condition = regint(1)
    instructions.crash(regint.conv(condition))

def public_input():
    """ Public input read from ``Programs/Public-Input/<progname>``. """
    res = cint()
    pubinput(res)
    return res

# mostly obsolete functions
# use the equivalent from types.py

@vectorize
def store_in_mem(value, address):
    if isinstance(value, int):
        value = regint(value)
    try:
        value.store_in_mem(address)
    except AttributeError:
        if isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                store_in_mem(x, address + i)
            return
        # legacy
        if value.is_clear:
            if isinstance(address, cint):
                stmci(value, address)
            else:
                stmc(value, address)
        else:
            if isinstance(address, cint):
                stmsi(value, address)
            else:
                stms(value, address)

@set_instruction_type
@vectorize
def reveal(secret):
    try:
        return secret.reveal()
    except AttributeError:
        if secret.is_clear:
            return secret
        if secret.is_gf2n:
            res = cgf2n()
        else:
            res = cint()
        instructions.asm_open(True, res, secret)
        return res

@vectorize
def get_thread_number():
    """ Returns the thread number. """
    res = regint()
    ldtn(res)
    return res

@vectorize
def get_arg():
    """ Returns the thread argument. """
    res = regint()
    ldarg(res)
    return res

def make_array(l, t=None):
    if isinstance(l, program.Tape.Register):
        res = Array(len(l), t or type(l))
        res[:] = l
    else:
        l = list(l)
        res = Array(len(l), t or type(l[0]) if l else cint)
        res.assign(l)
    return res


class FunctionTapeCall:
    def __init__(self, thread, base, bases):
        self.thread = thread
        self.base = base
        self.bases = bases
    def start(self):
        self.thread.start(self.base)
        return self
    def join(self):
        self.thread.join()
        instructions.program.free(self.base, 'ci')
        for reg_type,addr in self.bases.items():
            get_program().free(addr, reg_type.reg_type)

class Function:
    def __init__(self, function, name=None, compile_args=[]):
        self.type_args = {}
        self.function = function
        self.name = name
        if name is None:
            self.name = self.function.__name__
        self.compile_args = compile_args
    def __call__(self, *args):
        args = tuple(arg.read() if isinstance(arg, MemValue) else arg for arg in args)
        from .types import _types
        get_reg_type = lambda x: \
            regint if isinstance(x, int) else _types.get(x.reg_type, type(x))
        if len(args) not in self.type_args:
            # first call
            type_args = collections.defaultdict(list)
            for i,arg in enumerate(args):
                type_args[get_reg_type(arg)].append(i)
            def wrapped_function(*compile_args):
                base = get_arg()  # obtain the start address of function parameters
                bases = dict((t, regint.load_mem(base + i)) \
                                 for i,t in enumerate(sorted(type_args,
                                                             key=lambda x:
                                                             x.reg_type))) # obtain the start addresses for different types of data
                runtime_args = [None] * len(args)
                for t in sorted(type_args, key=lambda x: x.reg_type):
                    i = 0
                    for i_arg in type_args[t]:
                        runtime_args[i_arg] = t.load_mem(bases[t] + i)
                        i += util.mem_size(t)
                return self.function(*(list(compile_args) + runtime_args))
            self.on_first_call(wrapped_function)
            self.type_args[len(args)] = type_args
        type_args = self.type_args[len(args)]
        base = instructions.program.malloc(len(type_args), 'ci')
        bases = dict((t, get_program().malloc(len(type_args[t]), t)) \
                         for t in type_args)
        for i,reg_type in enumerate(sorted(type_args,
                                           key=lambda x: x.reg_type)):
            store_in_mem(bases[reg_type], base + i)
            j = 0
            for i_arg in type_args[reg_type]:
                if get_reg_type(args[i_arg]) != reg_type:
                    raise CompilerError('type mismatch: "%s" not of type "%s"' %
                                        (args[i_arg], reg_type))
                store_in_mem(args[i_arg], bases[reg_type] + j)
                j += util.mem_size(reg_type)
        return self.on_call(base, bases)

class FunctionTape(Function):
    # not thread-safe
    def __init__(self, function, name=None, compile_args=[],
                 single_thread=False):
        Function.__init__(self, function, name, compile_args)
        self.single_thread = single_thread
    def on_first_call(self, wrapped_function):
        self.thread = MPCThread(wrapped_function, self.name,
                                args=self.compile_args,
                                single_thread=self.single_thread)
    def on_call(self, base, bases):
        return FunctionTapeCall(self.thread, base, bases)

def function_tape(function):
    return FunctionTape(function)

def function_tape_with_compile_args(*args):
    def wrapper(function):
        return FunctionTape(function, compile_args=args)
    return wrapper

def single_thread_function_tape(function):
    return FunctionTape(function, single_thread=True)

def memorize(x):
    if isinstance(x, (tuple, list)):
        return tuple(memorize(i) for i in x)
    else:
        return MemValue(x)

def unmemorize(x):
    if isinstance(x, (tuple, list)):
        return tuple(unmemorize(i) for i in x)
    else:
        return x.read()

class FunctionBlock(Function):
    def on_first_call(self, wrapped_function):
        old_block = get_tape().active_basicblock
        parent_node = get_tape().req_node
        get_tape().open_scope(lambda x: x[0], None, 'begin-' + self.name)
        block = get_tape().active_basicblock
        block.alloc_pool = defaultdict(list)
        del parent_node.children[-1]
        self.node = get_tape().req_node
        if get_program().verbose:
            print('Compiling function', self.name)
        result = wrapped_function(*self.compile_args)
        if result is not None:
            self.result = memorize(result)
        else:
            self.result = None
        if get_program().verbose:
            print('Done compiling function', self.name)
        p_return_address = get_tape().program.malloc(1, 'ci')
        get_tape().function_basicblocks[block] = p_return_address
        return_address = regint.load_mem(p_return_address)
        get_tape().active_basicblock.set_exit(instructions.jmpi(return_address, add_to_prog=False))
        self.last_sub_block = get_tape().active_basicblock
        get_tape().close_scope(old_block, parent_node, 'end-' + self.name)
        old_block.set_exit(instructions.jmp(0, add_to_prog=False), get_tape().active_basicblock)
        self.basic_block = block

    def on_call(self, base, bases):
        if base is not None:
            instructions.starg(regint(base))
        block = self.basic_block
        if block not in get_tape().function_basicblocks:
            raise CompilerError('unknown function')
        old_block = get_tape().active_basicblock
        old_block.set_exit(instructions.jmp(0, add_to_prog=False), block)
        p_return_address = get_tape().function_basicblocks[block]
        return_address = get_tape().new_reg('ci')
        old_block.return_address_store = instructions.ldint(return_address, 0)
        instructions.stmint(return_address, p_return_address)
        get_tape().start_new_basicblock(name='call-' + self.name)
        get_tape().active_basicblock.set_return(old_block, self.last_sub_block)
        get_tape().req_node.children.append(self.node)
        if self.result is not None:
            return unmemorize(self.result)

def function_block(function):
    return FunctionBlock(function)

def function_block_with_compile_args(*args):
    def wrapper(function):
        return FunctionBlock(function, compile_args=args)
    return wrapper

def method_block(function):
    # If you use this, make sure to use MemValue for all member
    # variables.
    compiled_functions = {}
    def wrapper(self, *args):
        if self in compiled_functions:
            return compiled_functions[self](*args)
        else:
            name = '%s-%s' % (type(self).__name__, function.__name__)
            block = FunctionBlock(function, name=name, compile_args=(self,))
            compiled_functions[self] = block
            return block(*args)
    return wrapper

def cond_swap(x,y):
    from .types import SubMultiArray
    if isinstance(x, (Array, SubMultiArray)):
        b = x[0] > y[0]
        return list(zip(*[b.cond_swap(xx, yy) for xx, yy in zip(x, y)]))
    b = x < y
    if isinstance(x, sfloat):
        res = ([], [])
        for i,j in enumerate(('v','p','z','s')):
            xx = x.__getattribute__(j)
            yy = y.__getattribute__(j)
            bx = b * xx
            by = b * yy
            res[0].append(bx + yy - by)
            res[1].append(xx - bx + by)
        return sfloat(*res[0]), sfloat(*res[1])
    return b.cond_swap(y, x)

def sort(a):
    res = a
    
    for i in range(len(a)):
        for j in reversed(list(range(i))):
            res[j], res[j+1] = cond_swap(res[j], res[j+1])

    return res

def odd_even_merge(a):
    if len(a) == 2:
        a[0], a[1] = cond_swap(a[0], a[1])
    else:
        even = a[::2]
        odd = a[1::2]
        odd_even_merge(even)
        odd_even_merge(odd)
        a[0] = even[0]
        for i in range(1, len(a) // 2):
            a[2*i-1], a[2*i] = cond_swap(odd[i-1], even[i])
        a[-1] = odd[-1]

def odd_even_merge_sort(a):
    if len(a) == 1:
        return
    elif len(a) % 2 == 0:
        aa = a
        a = list(a)
        lower = a[:len(a)//2]
        upper = a[len(a)//2:]
        odd_even_merge_sort(lower)
        odd_even_merge_sort(upper)
        a[:] = lower + upper
        odd_even_merge(a)
        aa[:] = a
    else:
        raise CompilerError('Length of list must be power of two')

def chunky_odd_even_merge_sort(a):
    tmp = a[0].Array(len(a))
    for i,j in enumerate(a):
        tmp[i] = j
    l = 1
    while l < len(a):
        l *= 2
        k = 1
        while k < l:
            k *= 2
            def round():
                for i in range(len(a)):
                    a[i] = tmp[i]
                for i in range(len(a) // l):
                    for j in range(l // k):
                        base = i * l + j
                        step = l // k
                        if k == 2:
                            a[base], a[base+step] = cond_swap(a[base], a[base+step])
                        else:
                            b = a[base:base+k*step:step]
                            for m in range(base + step, base + (k - 1) * step, 2 * step):
                                a[m], a[m+step] = cond_swap(a[m], a[m+step])
                for i in range(len(a)):
                    tmp[i] = a[i]
            chunk = MPCThread(round, 'sort-%d-%d' % (l,k), single_thread=True)
            chunk.start()
            chunk.join()
            #round()
    for i in range(len(a)):
        a[i] = tmp[i]

def chunkier_odd_even_merge_sort(a, n=None, max_chunk_size=512, n_threads=7, use_chunk_wraps=False):
    if n is None:
        n = len(a)
        a_base = instructions.program.malloc(n, 's')
        for i,j in enumerate(a):
            store_in_mem(j, a_base + i)
    else:
        a_base = a
    tmp_base = instructions.program.malloc(n, 's')
    chunks = {}
    threads = []

    def run_threads():
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        del threads[:]

    def run_chunk(size, base):
        if size not in chunks:
            def swap_list(list_base):
                for i in range(size // 2):
                    base = list_base + 2 * i
                    x, y = cond_swap(sint.load_mem(base),
                                     sint.load_mem(base + 1))
                    store_in_mem(x, base)
                    store_in_mem(y, base + 1)
            chunks[size] = FunctionTape(swap_list, 'sort-%d' % size)
        return chunks[size](base)

    def run_round(size):
        # minimize number of chunk sizes
        n_chunks = int(math.ceil(1.0 * size / max_chunk_size))
        lower_size = size // n_chunks // 2 * 2
        n_lower_size = n_chunks - (size - n_chunks * lower_size) // 2
        # print len(to_swap) == lower_size * n_lower_size + \
        #     (lower_size + 2) * (n_chunks - n_lower_size), \
        #     len(to_swap), n_chunks, lower_size, n_lower_size
        base = 0
        round_threads = []
        for i in range(n_lower_size):
            round_threads.append(run_chunk(lower_size, tmp_base + base))
            base += lower_size
        for i in range(n_chunks - n_lower_size):
            round_threads.append(run_chunk(lower_size + 2, tmp_base + base))
            base += lower_size + 2
        run_threads_in_rounds(round_threads)

    postproc_chunks = []
    wrap_chunks = {}
    post_threads = []
    pre_threads = []

    def load_and_store(x, y, to_right):
        if to_right:
            store_in_mem(sint.load_mem(x), y)
        else:
            store_in_mem(sint.load_mem(y), x)

    def run_setup(k, a_addr, step, tmp_addr):
        if k == 2:
            def mem_op(preproc, a_addr, step, tmp_addr):
                load_and_store(a_addr, tmp_addr, preproc)
                load_and_store(a_addr + step, tmp_addr + 1, preproc)
            res = 2
        else:
            def mem_op(preproc, a_addr, step, tmp_addr):
                instructions.program.curr_tape.merge_opens = False
#                for i,m in enumerate(range(a_addr + step, a_addr + (k - 1) * step, step)):
                for i in range(k - 2):
                    m = a_addr + step + i * step
                    load_and_store(m, tmp_addr + i, preproc)
            res = k - 2
        if not use_chunk_wraps or k <= 4:
            mem_op(True, a_addr, step, tmp_addr)
            postproc_chunks.append((mem_op, (a_addr, step, tmp_addr)))
        else:
            if k not in wrap_chunks:
                pre_chunk = FunctionTape(mem_op, 'pre-%d' % k,
                                         compile_args=[True])
                post_chunk = FunctionTape(mem_op, 'post-%d' % k,
                                          compile_args=[False])
                wrap_chunks[k] = (pre_chunk, post_chunk)
            pre_chunk, post_chunk = wrap_chunks[k]
            pre_threads.append(pre_chunk(a_addr, step, tmp_addr))
            post_threads.append(post_chunk(a_addr, step, tmp_addr))
        return res

    def run_threads_in_rounds(all_threads):
        for thread in all_threads:
            if len(threads) == n_threads:
                run_threads()
            threads.append(thread)
        run_threads()
        del all_threads[:]

    def run_postproc():
        run_threads_in_rounds(post_threads)
        for chunk,args in postproc_chunks:
            chunk(False, *args)
        postproc_chunks[:] = []

    l = 1
    while l < n:
        l *= 2
        k = 1
        while k < l:
            k *= 2
            size = 0
            instructions.program.curr_tape.merge_opens = False
            for i in range(n // l):
                for j in range(l // k):
                    base = i * l + j
                    step = l // k
                    size += run_setup(k, a_base + base, step, tmp_base + size)
            run_threads_in_rounds(pre_threads)
            run_round(size)
            run_postproc()

    if isinstance(a, list):
        for i in range(n):
            a[i] = sint.load_mem(a_base + i)
        instructions.program.free(a_base, 's')
    instructions.program.free(tmp_base, 's')

def loopy_chunkier_odd_even_merge_sort(a, n=None, max_chunk_size=512, n_threads=7):
    if n is None:
        n = len(a)
        a_base = instructions.program.malloc(n, 's')
        for i,j in enumerate(a):
            store_in_mem(j, a_base + i)
    else:
        a_base = a
    tmp_base = instructions.program.malloc(n, 's')
    tmp_i = instructions.program.malloc(1, 'ci')
    chunks = {}
    threads = []

    def run_threads():
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        del threads[:]

    def run_threads_in_rounds(all_threads):
        for thread in all_threads:
            if len(threads) == n_threads:
                run_threads()
            threads.append(thread)
        run_threads()
        del all_threads[:]

    def run_chunk(size, base):
        if size not in chunks:
            def swap_list(list_base):
                for i in range(size // 2):
                    base = list_base + 2 * i
                    x, y = cond_swap(sint.load_mem(base),
                                     sint.load_mem(base + 1))
                    store_in_mem(x, base)
                    store_in_mem(y, base + 1)
            chunks[size] = FunctionTape(swap_list, 'sort-%d' % size)
        return chunks[size](base)

    def run_round(size):
        # minimize number of chunk sizes
        n_chunks = int(math.ceil(1.0 * size / max_chunk_size))
        lower_size = size // n_chunks // 2 * 2
        n_lower_size = n_chunks - (size - n_chunks * lower_size) // 2
        # print len(to_swap) == lower_size * n_lower_size + \
        #     (lower_size + 2) * (n_chunks - n_lower_size), \
        #     len(to_swap), n_chunks, lower_size, n_lower_size
        base = 0
        round_threads = []
        for i in range(n_lower_size):
            round_threads.append(run_chunk(lower_size, tmp_base + base))
            base += lower_size
        for i in range(n_chunks - n_lower_size):
            round_threads.append(run_chunk(lower_size + 2, tmp_base + base))
            base += lower_size + 2
        run_threads_in_rounds(round_threads)

    l = 1
    while l < n:
        l *= 2
        k = 1
        while k < l:
            k *= 2
            def load_and_store(x, y):
                if to_tmp:
                    store_in_mem(sint.load_mem(x), y)
                else:
                    store_in_mem(sint.load_mem(y), x)
            def outer(i):
                def inner(j):
                    base = j + a_base + i * l
                    step = l // k
                    if k == 2:
                        tmp_addr = regint.load_mem(tmp_i)
                        load_and_store(base, tmp_addr)
                        load_and_store(base + step, tmp_addr + 1)
                        store_in_mem(tmp_addr + 2, tmp_i)
                    else:
                        def inner2(m):
                            m += base
                            tmp_addr = regint.load_mem(tmp_i)
                            load_and_store(m, tmp_addr)
                            store_in_mem(tmp_addr + 1, tmp_i)
                        range_loop(inner2, step, (k - 1) * step, step)
                range_loop(inner, l // k)
            instructions.program.curr_tape.merge_opens = False
            to_tmp = True
            store_in_mem(tmp_base, tmp_i)
            range_loop(outer, n // l)
            if k == 2:
                run_round(n)
            else:
                run_round(n // k * (k - 2))
            instructions.program.curr_tape.merge_opens = False
            to_tmp = False
            store_in_mem(tmp_base, tmp_i)
            range_loop(outer, n // l)

    if isinstance(a, list):
        for i in range(n):
            a[i] = sint.load_mem(a_base + i)
        instructions.program.free(a_base, 's')
    instructions.program.free(tmp_base, 's')
    instructions.program.free(tmp_i, 'ci')


def loopy_odd_even_merge_sort(a, sorted_length=1, n_parallel=32,
                              n_threads=None):
    steps = {}
    l = sorted_length
    while l < len(a):
        l *= 2
        k = 1
        while k < l:
            k *= 2
            n_innermost = 1 if k == 2 else k // 2 - 1
            key = k
            if key not in steps:
                @function_block
                def step(l):
                    l = MemValue(l)
                    m = 2 ** int(math.ceil(math.log(len(a), 2)))
                    @for_range_opt_multithread(n_threads, m // k)
                    def _(i):
                        n_inner = l // k
                        j = i % n_inner
                        i //= n_inner
                        base = i*l + j
                        step = l//k
                        def swap(base, step):
                            if m == len(a):
                                a[base], a[base + step] = \
                                    cond_swap(a[base], a[base + step])
                            else:
                                # ignore values outside range
                                go = base + step < len(a)
                                x = a.maybe_get(go, base)
                                y = a.maybe_get(go, base + step)
                                tmp = cond_swap(x, y)
                                for i, idx in enumerate((base, base + step)):
                                    a.maybe_set(go, idx, tmp[i])
                        if k == 2:
                            swap(base, step)
                        else:
                            @for_range_opt(n_innermost)
                            def f(i):
                                m1 = step + i * 2 * step
                                m2 = m1 + base
                                swap(m2, step)
                steps[key] = step
            steps[key](l)

def mergesort(A):
    B = Array(len(A), sint)

    def merge(i_left, i_right, i_end):
        i0 = MemValue(i_left)
        i1 = MemValue(i_right)
        @for_range(i_left, i_end)
        def loop(j):
            if_then(and_(lambda: i0 < i_right,
                         or_(lambda: i1 >= i_end,
                             lambda: regint(reveal(A[i0] <= A[i1])))))
            B[j] = A[i0]
            i0.iadd(1)
            else_then()
            B[j] = A[i1]
            i1.iadd(1)
            end_if()

    width = MemValue(1)
    @do_while
    def width_loop():
        @for_range(0, len(A), 2 * width)
        def merge_loop(i):
            merge(i, i + width, i + 2 * width)
        A.assign(B)
        width.imul(2)
        return width < len(A)

def range_loop(loop_body, start, stop=None, step=None):
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    def loop_fn(i):
        res = loop_body(i)
        return util.if_else(res == 0, stop, i + step)
    if isinstance(step, int):
        if step > 0:
            condition = lambda x: x < stop
        elif step < 0:
            condition = lambda x: x > stop
        else:
            raise CompilerError('step must not be zero')
    else:
        b = step > 0
        condition = lambda x: b * (x < stop) + (1 - b) * (x > stop)
    while_loop(loop_fn, condition, start, g=loop_body.__globals__)
    if isinstance(start, int) and isinstance(stop, int) \
            and isinstance(step, int):
        # known loop count
        if condition(start):
            get_tape().req_node.children[-1].aggregator = \
                lambda x: int(ceil(((stop - start) / step))) * x[0]

def for_range(start, stop=None, step=None):
    """
    Decorator to execute loop bodies consecutively.  Arguments work as
    in Python :py:func:`range`, but they can be any public
    integer. Information has to be passed out via container types such
    as :py:class:`~Compiler.types.Array` or using :py:func:`update`.
    Note that changing Python data structures such
    as lists within the loop is not possible, but the compiler cannot
    warn about this.

    :param start/stop/step: regint/cint/int

    Example:

    .. code::

        a = sint.Array(n)
        x = sint(0)
        @for_range(n)
        def _(i):
            a[i] = i
            x.update(x + 1)

    Note that you cannot overwrite data structures such as
    :py:class:`~Compiler.types.Array` in a loop.  Use
    :py:func:`~Compiler.types.Array.assign` instead.
    """
    def decorator(loop_body):
        range_loop(loop_body, start, stop, step)
        return loop_body
    return decorator

def for_range_parallel(n_parallel, n_loops):
    """
    Decorator to execute a loop :py:obj:`n_loops` up to
    :py:obj:`n_parallel` loop bodies in parallel.
    Using any other control flow instruction inside the loop breaks
    the optimization.

    :param n_parallel: compile-time (int)
    :param n_loops: regint/cint/int or list of int

    Example:

    .. code::

        @for_range_parallel(n_parallel, n_loops)
        def _(i):
            a[i] = a[i] * a[i]

    Multidimensional ranges are supported as well. The following
    executes ``f(0, 0)`` to ``f(4, 2)``, two calls in parallel.

    .. code::

        @for_range_parallel(2, [5, 3])
        def f(i, j):
            ...
    """
    if isinstance(n_loops, (list, tuple)):
        return for_range_multithread(None, n_parallel, n_loops)
    return map_reduce_single(n_parallel, n_loops)

def for_range_opt(n_loops, budget=None):
    """ Execute loop bodies in parallel up to an optimization budget.
    This prevents excessive loop unrolling. The budget is respected
    even with nested loops. Note that the optimization is rather
    rudimentary for runtime :py:obj:`n_loops` (regint/cint). Consider
    using :py:func:`for_range_parallel` in this case.
    Using further control flow constructions inside other than
    :py:func:`for_range_opt` (e.g, :py:func:`for_range`) breaks the
    optimization.

    :param n_loops: int/regint/cint
    :param budget: number of instructions after which to start optimization (default is 100,000)

    Example:

    .. code::

        @for_range_opt(n)
        def _(i):
            ...

    Multidimensional ranges are supported as well. The following
    executes ``f(0, 0)`` to ``f(4, 2)`` in parallel according to
    the budget.

    .. code::

        @for_range_opt([5, 3])
        def f(i, j):
            ...
    """
    if isinstance(n_loops, (list, tuple)):
        return for_range_opt_multithread(None, n_loops)
    return map_reduce_single(None, n_loops, budget=budget)

def map_reduce_single(n_parallel, n_loops, initializer=lambda *x: [],
                      reducer=lambda *x: [], mem_state=None, budget=None):
    budget = budget or get_program().budget
    if not (isinstance(n_parallel, int) or n_parallel is None):
        raise CompilerError('Number of parallel executions must be constant')
    n_parallel = 1 if is_zero(n_parallel) else n_parallel
    if mem_state is None:
        # default to list of MemValues to allow varying types
        mem_state = [MemValue(x) for x in initializer()]
        use_array = False
    else:
        # use Arrays for multithread version
        use_array = True
    if not util.is_constant(n_loops):
        budget //= 10
        n_loops = regint(n_loops)
    def decorator(loop_body):
        my_n_parallel = n_parallel
        if isinstance(n_parallel, int):
            if isinstance(n_loops, int):
                loop_rounds = n_loops // n_parallel \
                              if n_parallel < n_loops else 0
            else:
                loop_rounds = n_loops / n_parallel
        def write_state_to_memory(r):
            if use_array:
                mem_state.assign(r)
            else:
                # cannot do mem_state = [...] due to scope issue
                for j,x in enumerate(r):
                    mem_state[j].write(x)
        if n_parallel is not None:
            # will be optimized out if n_loops <= n_parallel
            @for_range(loop_rounds)
            def f(i):
                state = tuplify(initializer())
                start_block = get_block()
                j = i * n_parallel
                one = regint(1)
                for k in range(n_parallel):
                    state = reducer(tuplify(loop_body(j)), state)
                    j += one
                if n_parallel > 1 and start_block != get_block():
                    print('WARNING: parallelization broken '
                          'by control flow instruction')
                r = reducer(mem_state, state)
                write_state_to_memory(r)
        else:
            if is_zero(n_loops):
                return
            n_opt_loops_reg = regint(0)
            n_opt_loops_inst = get_block().instructions[-1]
            parent_block = get_block()
            @while_do(lambda x: x + n_opt_loops_reg <= n_loops, regint(0))
            def _(i):
                state = tuplify(initializer())
                k = 0
                block = get_block()
                assert not isinstance(n_loops, int) or n_loops > 0
                pre = copy.copy(loop_body.__globals__)
                while (not util.is_constant(n_loops) or k < n_loops) \
                      and (len(get_block()) < budget or k == 0) \
                      and block is get_block():
                    j = i + k
                    state = reducer(tuplify(loop_body(j)), state)
                    k += 1
                _link(pre, loop_body.__globals__)
                r = reducer(mem_state, state)
                write_state_to_memory(r)
                global n_opt_loops
                n_opt_loops = k
                n_opt_loops_inst.args[1] = k
                return i + k
            my_n_parallel = n_opt_loops
            loop_rounds = n_loops // my_n_parallel
            blocks = get_tape().basicblocks
            n_to_merge = 5
            if util.is_one(loop_rounds) and parent_block is blocks[-n_to_merge]:
                # merge blocks started by if and do_while
                def exit_elimination(block):
                    if block.exit_condition is not None:
                        for reg in block.exit_condition.get_used():
                            reg.can_eliminate = True
                exit_elimination(parent_block)
                merged = parent_block
                merged.exit_condition = blocks[-1].exit_condition
                merged.exit_block = blocks[-1].exit_block
                assert parent_block is blocks[-n_to_merge]
                assert blocks[-n_to_merge + 1] is \
                    get_tape().req_node.children[-1].nodes[0].blocks[0]
                for block in blocks[-n_to_merge + 1:]:
                    merged.instructions += block.instructions
                    exit_elimination(block)
                    block.purge(retain_usage=False)
                del blocks[-n_to_merge + 1:]
                del get_tape().req_node.children[-1]
                merged.children = []
                RegintOptimizer().run(merged.instructions)
                get_tape().active_basicblock = merged
            else:
                req_node = get_tape().req_node.children[-1].nodes[0]
                if util.is_constant(loop_rounds):
                    req_node.children[0].aggregator = lambda x: loop_rounds * x[0]
        if isinstance(n_loops, int):
            state = mem_state
            for j in range(loop_rounds * my_n_parallel, n_loops):
                state = reducer(tuplify(loop_body(j)), state)
        else:
            @for_range(loop_rounds * my_n_parallel, n_loops)
            def f(j):
                r = reducer(tuplify(loop_body(j)), mem_state)
                write_state_to_memory(r)
            state = mem_state
        for i,x in enumerate(state):
            if use_array:
                mem_state[i] = x
            else:
                mem_state[i].write(x)
        def returner():
            return untuplify(tuple(state))
        return returner
    return decorator

def for_range_multithread(n_threads, n_parallel, n_loops, thread_mem_req={}):
    """
    Execute :py:obj:`n_loops` loop bodies in up to :py:obj:`n_threads`
    threads, up to :py:obj:`n_parallel` in parallel per thread.

    :param n_threads/n_parallel: compile-time (int)
    :param n_loops: regint/cint/int

    """
    return map_reduce(n_threads, n_parallel, n_loops, \
                          lambda *x: [], lambda *x: [], thread_mem_req)

def for_range_opt_multithread(n_threads, n_loops):
    """
    Execute :py:obj:`n_loops` loop bodies in up to :py:obj:`n_threads`
    threads, in parallel up to an optimization budget per thread
    similar to :py:func:`for_range_opt`. Note that optimization is rather
    rudimentary for runtime :py:obj:`n_loops` (regint/cint). Consider
    using :py:func:`for_range_multithread` in this case.

    :param n_threads: compile-time (int)
    :param n_loops: regint/cint/int

    The following will execute loop bodies 0-9 in one thread, 10-19 in
    another etc:

    .. code::

        @for_range_opt_multithread(8, 80)
        def _(i):
            ...

    Multidimensional ranges are supported as well. The following
    executes ``f(0, 0)`` to ``f(2, 0)`` in one thread and ``f(2, 1)``
    to ``f(4, 2)`` in another.

    .. code::

        @for_range_opt_multithread(2, [5, 3])
        def f(i, j):
            ...
    """
    return for_range_multithread(n_threads, None, n_loops)

def multithread(n_threads, n_items=None, max_size=None):
    """
    Distribute the computation of :py:obj:`n_items` to
    :py:obj:`n_threads` threads, but leave the in-thread repetition up
    to the user.

    :param n_threads: compile-time (int)
    :param n_items: regint/cint/int (default: :py:obj:`n_threads`)
    :param max_size: maximum size to be processed at once (default: no limit)

    The following executes ``f(0, 8)``, ``f(8, 8)``, and
    ``f(16, 9)`` in three different threads:

    .. code::

        @multithread(8, 25)
        def f(base, size):
            ...
    """
    if n_items is None:
        n_items = n_threads
    if max_size is None or n_items <= max_size:
        return map_reduce(n_threads, None, n_items, initializer=lambda: [],
                          reducer=None, looping=False)
    else:
        def wrapper(function):
            @multithread(n_threads, n_items)
            def new_function(base, size):
                @for_range(size // max_size)
                def _(i):
                    function(base + i * max_size, max_size)
                rem = size % max_size
                if rem:
                    function(base + size - rem, rem)
        return wrapper

def map_reduce(n_threads, n_parallel, n_loops, initializer, reducer, \
                   thread_mem_req={}, looping=True):
    assert(n_threads != 0)
    if isinstance(n_loops, (list, tuple)): # handle cases where n_loops is list or tuple
        split = n_loops
        n_loops = reduce(operator.mul, n_loops)
        def decorator(loop_body):
            def new_body(i):
                indices = []
                for n in reversed(split):
                    indices.insert(0, i % n)
                    i //= n
                return loop_body(*indices)
            return new_body
        new_dec = map_reduce(n_threads, n_parallel, n_loops, initializer, reducer, thread_mem_req)
        return lambda loop_body: new_dec(decorator(loop_body))
    n_loops = MemValue.if_necessary(n_loops) # handle the case where n_loops is int
    if n_threads == None or util.is_one(n_loops): # handle the case where n_threads or n_loops is 1
        if not looping:
            return lambda loop_body: loop_body(0, n_loops)
        dec = map_reduce_single(n_parallel, n_loops, initializer, reducer)
        if thread_mem_req:
            thread_mem = Array(thread_mem_req[regint], regint)
            return lambda loop_body: dec(lambda i: loop_body(i, thread_mem))
        else:
            return dec
    def decorator(loop_body): # handle the case where n_loops is more than 1
        thread_rounds = MemValue.if_necessary(n_loops // n_threads)
        if util.is_constant(thread_rounds):
            remainder = n_loops % n_threads
        else:
            remainder = 0
        for t in thread_mem_req:
            if t != regint:
                raise CompilerError('Not implemented for other than regint')
        args = Matrix(n_threads, 2 + thread_mem_req.get(regint, 0), 'ci')# create args for each thread
        state = initializer()
        if len(state) == 0:
            state_type = cint
        elif isinstance(state, (tuple, list)):
            state_type = type(state[0])
        else:
            state_type = type(state) # decide the type of state
        def f(inc): # the function run by each thread
            base = args[get_arg()][0]
            if not util.is_constant(thread_rounds):
                i = base / thread_rounds
                overhang = n_loops % n_threads
                inc = i < overhang
                base += inc.if_else(i, overhang)
            if not looping:
                return loop_body(base, thread_rounds + inc)
            if thread_mem_req:
                thread_mem = Array(thread_mem_req[regint], regint, \
                                       args[get_arg()].address + 2)
            mem_state = Array(len(state), state_type, args[get_arg()][1])
            @map_reduce_single(n_parallel, thread_rounds + inc, \
                                   initializer, reducer, mem_state)
            def f(i):
                if thread_mem_req:
                    return loop_body(base + i, thread_mem)  # might be the new_body from the upper level call
                else:
                    return loop_body(base + i)
        prog = get_program()
        thread_args = []
        if prog.curr_tape == prog.tapes[0]:
            prog.n_running_threads = n_threads
        if not util.is_zero(thread_rounds):
            tape = prog.new_tape(f, (0,), 'multithread')  # creating the new tape to be run
            for i in range(n_threads - remainder):
                mem_state = make_array(initializer())
                args[remainder + i][0] = i * thread_rounds
                if len(mem_state):
                    args[remainder + i][1] = mem_state.address
                thread_args.append((tape, remainder + i))
        if remainder:
            tape1 = prog.new_tape(f, (1,), 'multithread1')
            for i in range(remainder):
                mem_state = make_array(initializer())
                args[i][0] = (n_threads - remainder + i) * thread_rounds + i
                if len(mem_state):
                    args[i][1] = mem_state.address
                thread_args.append((tape1, i))
        prog.n_running_threads = None
        threads = prog.run_tapes(thread_args)
        for thread in threads:
            prog.join_tape(thread)
        if len(state):
            if thread_rounds:
                for i in range(n_threads - remainder):
                    state = reducer(Array(len(state), state_type, \
                                              args[remainder + i][1]), state)
            if remainder:
                for i in range(remainder):
                    state = reducer(Array(len(state), state_type, \
                                              args[i][1]), state)
        def returner():
            return untuplify(state)
        return returner
    return decorator

def build_block():
    return 1

def map_sum(n_threads, n_parallel, n_loops, n_items, value_types):
    value_types = tuplify(value_types)
    if len(value_types) == 1:
        value_types *= n_items
    elif len(value_types) != n_items:
        raise CompilerError('Incorrect number of value_types.')
    initializer = lambda: [t(0) for t in value_types]
    def summer(x,y):
        return tuple(a + b for a,b in zip(x,y))
    return map_reduce(n_threads, n_parallel, n_loops, initializer, summer)

def map_sum_opt(n_threads, n_loops, types):
    """ Multi-threaded sum reduction. The following computes a sum of
    ten squares in three threads::

        @map_sum_opt(3, 10, [sint])
        def summer(i):
            return sint(i) ** 2

        result = summer()

    :param n_threads: number of threads (int)
    :param n_loops: number of loop runs (regint/cint/int)
    :param types: return type, must match the return statement
        in the loop

    """
    return map_sum(n_threads, None, n_loops, len(types), types)

def map_sum_simple(n_threads, n_loops, type, size):
    """ Vectorized multi-threaded sum reduction. The following computes a
    100 sums of ten squares in three threads::

        @map_sum_simple(3, 10, sint, 100)
        def summer(i):
            return sint(regint.inc(100, i, 0)) ** 2

        result = summer()

    :param n_threads: number of threads (int)
    :param n_loops: number of loop runs (regint/cint/int)
    :param type: return type, must match the return statement
        in the loop
    :param size: vector size, must match the return statement
        in the loop

    """
    initializer = lambda: type(0, size=size)
    def summer(*args):
        assert len(args) == 2
        args = list(args)
        for i in (0, 1):
            if isinstance(args[i], tuple):
                assert len(args[i]) == 1
                args[i] = args[i][0]
        for i in (0, 1):
            assert len(args[i]) == size
            if isinstance(args[i], Array):
                args[i] = args[i][:]
        return args[0] + args[1]
    return map_reduce(n_threads, 1, n_loops, initializer, summer)

def tree_reduce_multithread(n_threads, function, vector):
    """ Round-efficient reduction in several threads. The following code
    computes the maximum of an array in 10 threads::

      tree_reduce_multithread(10, lambda x, y: x.max(y), a)

    :param n_threads: number of threads (int)
    :param function: reduction function taking exactly two arguments
    :param vector: register vector or array

    """
    inputs = vector.Array(len(vector))
    inputs.assign_vector(vector)
    outputs = vector.Array(len(vector) // 2)
    left = len(vector)
    while left > 1:
        @multithread(n_threads, left // 2)
        def _(base, size):
            outputs.assign_vector(
                function(inputs.get_vector(2 * base, size),
                         inputs.get_vector(2 * base + size, size)), base)
        inputs.assign_vector(outputs.get_vector(0, left // 2))
        if left % 2 == 1:
            inputs[left // 2] = inputs[left - 1]
        left = (left + 1) // 2
    return inputs[0]

def tree_reduce(function, sequence):
    """ Round-efficient reduction. The following computes the maximum
    of the list :py:obj:`l`::

      m = tree_reduce(lambda x, y: x.max(y), l)

    :param function: reduction function taking two arguments
    :param sequence: list, vector, or array

    """
    return util.tree_reduce(function, sequence)

def foreach_enumerate(a):
    """ Run-time loop over public data. This uses
    ``Player-Data/Public-Input/<progname>``. Example:

    .. code::

        @foreach_enumerate([2, 8, 3])
        def _(i, j):
            print_ln('%s: %s', i, j)

    This will output:

    .. code::

        0: 2
        1: 8
        2: 3
    """
    for x in a:
        get_program().public_input(' '.join(str(y) for y in tuplify(x)))
    def decorator(loop_body):
        @for_range(len(a))
        def f(i):
            loop_body(i, *(public_input() for j in range(len(tuplify(a[0])))))
        return f
    return decorator

def while_loop(loop_body, condition, arg, g=None):
    if not callable(condition):
        raise CompilerError('Condition must be callable')
    # store arg in stack
    pre_condition = condition(arg)
    if not isinstance(pre_condition, (bool,int)) or pre_condition:
        arg = regint(arg)
        def loop_fn():
            result = loop_body(arg)
            if isinstance(result, MemValue):
                result = result.read()
            result.link(arg)
            cont = condition(result)
            return cont
        if_statement(pre_condition, lambda: do_while(loop_fn, g=g))

def while_do(condition, *args):
    """ While-do loop. The decorator requires an initialization, and
    the loop body function must return a suitable input for
    :py:obj:`condition`.

    :param condition: function returning public integer (regint/cint/int)
    :param args: arguments given to :py:obj:`condition` and loop body

    The following executes an ten-fold loop:

    .. code::

        @while_do(lambda x: x < 10, regint(0))
        def f(i):
            ...
            return i + 1
    """
    def decorator(loop_body):
        while_loop(loop_body, condition, *args)
        return loop_body
    return decorator

def do_loop(condition, loop_fn):
    # store initial condition to stack
    pushint(condition if isinstance(condition,regint) else regint(condition))
    def wrapped_loop():
        # save condition to stack
        new_cond = regint.pop()
        # run the loop
        condition = loop_fn(new_cond)
        pushint(condition)
        return condition
    do_while(wrapped_loop)
    regint.pop()

def _run_and_link(function, g=None):
    if g is None:
        g = function.__globals__
    pre = copy.copy(g)
    res = function()
    _link(pre, g)
    return res

def _link(pre, g):
    if g:
        from .types import _single
        for name, var in pre.items():
            if isinstance(var, (program.Tape.Register, _single, _vec)):
                new_var = g[name]
                if util.is_constant_float(new_var):
                    raise CompilerError('cannot reassign constants in blocks')
                if id(new_var) != id(var):
                    new_var.link(var)

def do_while(loop_fn, g=None):
    """ Do-while loop. The loop is stopped if the return value is zero.
    It must be public. The following executes exactly once:

    .. code::

        @do_while
        def _():
            ...
            return regint(0)
    """
    scope = instructions.program.curr_block
    parent_node = get_tape().req_node
    # possibly unknown loop count
    get_tape().open_scope(lambda x: x[0].set_all(float('Inf')), \
                              name='begin-loop')
    get_tape().loop_breaks.append([])
    loop_block = instructions.program.curr_block
    condition = _run_and_link(loop_fn, g)
    if callable(condition):
        condition = condition()
    branch = instructions.jmpnz(regint.conv(condition), 0, add_to_prog=False)
    instructions.program.curr_block.set_exit(branch, loop_block)
    get_tape().close_scope(scope, parent_node, 'end-loop')
    for loop_break in get_tape().loop_breaks.pop():
        loop_break.set_exit(jmp(0, add_to_prog=False), get_block())
    return loop_fn

def break_loop():
    """ Break out of loop. """
    get_tape().loop_breaks[-1].append(get_block())
    break_point('break')

def if_then(condition):
    class State: pass
    state = State()
    if callable(condition):
        condition = condition()
    try:
        if not condition.is_clear:
            raise CompilerError('cannot branch on secret values')
    except AttributeError:
        pass
    state.condition = regint.conv(condition)
    state.start_block = instructions.program.curr_block
    state.req_child = get_tape().open_scope(lambda x: x[0].max(x[1]), \
                                                   name='if-block')
    state.has_else = False
    state.caller = [frame[1:] for frame in inspect.stack()[1:]]
    instructions.program.curr_tape.if_states.append(state)

def else_then():
    try:
        state = instructions.program.curr_tape.if_states[-1]
    except IndexError:
        raise CompilerError('No open if block')
    if state.has_else:
        raise CompilerError('else block already defined')
    # run the else block
    state.if_exit_block = instructions.program.curr_block
    state.req_child.add_node(get_tape(), 'else-block')
    instructions.program.curr_tape.start_new_basicblock(state.start_block, \
                                                            name='else-block')
    state.else_block = instructions.program.curr_block
    state.has_else = True

def end_if():
    try:
        state = instructions.program.curr_tape.if_states.pop()
    except IndexError:
        raise CompilerError('No open if/else block')
    branch = instructions.jmpeqz(regint.conv(state.condition), 0, \
                                     add_to_prog=False)
    # start next block
    get_tape().close_scope(state.start_block, state.req_child.parent, 'end-if')
    if state.has_else:
        # jump to else block if condition == 0
        state.start_block.set_exit(branch, state.else_block)
        # set if block to skip else
        jump = instructions.jmp(0, add_to_prog=False)
        state.if_exit_block.set_exit(jump, instructions.program.curr_block)
    else:
        # set start block's conditional jump to next block
        state.start_block.set_exit(branch, instructions.program.curr_block)
        # nothing to compute without else
        state.req_child.aggregator = lambda x: x[0]

def if_statement(condition, if_fn, else_fn=None):
    if condition is True or condition is False:
        # condition known at compile time
        if condition:
            if_fn()
        elif else_fn is not None:
            else_fn()
    else:
        state = if_then(condition)
        if_fn()
        if else_fn is not None:
            else_then()
            else_fn()
        end_if()

def if_(condition):
    """
    Conditional execution without else block.

    :param condition: regint/cint/int

    Usage:

    .. code::

        @if_(x > 0)
        def _():
            ...
    """
    try:
        condition = bool(condition)
    except:
        pass
    def decorator(body):
        if isinstance(condition, bool):
            if condition:
                _run_and_link(body)
        else:
            if_then(condition)
            _run_and_link(body)
            end_if()
    return decorator

def if_e(condition):
    """
    Conditional execution with else block.
    Use :py:class:`~Compiler.types.MemValue` to assign values that
    live beyond.

    :param condition: regint/cint/int

    Usage:

    .. code::

        y = MemValue(0)
        @if_e(x > 0)
        def _():
            y.write(1)
        @else_
        def _():
            y.write(0)
    """
    try:
        condition = bool(condition)
    except:
        pass
    def decorator(body):
        if isinstance(condition, bool):
            get_tape().if_states.append(condition)
            if condition:
                _run_and_link(body)
        else:
            if_then(condition)
            _run_and_link(body)
    return decorator

def else_(body):
    if_states = get_tape().if_states
    if isinstance(if_states[-1], bool):
        if not if_states[-1]:
            _run_and_link(body)
        if_states.pop()
    else:
        else_then()
        _run_and_link(body)
        end_if()

def and_(*terms):
    res = regint(0)
    for term in terms:
        if_then(term())
    old_res = res
    res = regint(1)
    res.link(old_res)
    for term in terms:
        else_then()
        end_if()
    def load_result():
        return res
    return load_result

def or_(*terms):
    res = regint(1)
    for term in terms:
        if_then(term())
        else_then()
    old_res = res
    res = regint(0)
    res.link(old_res)
    for term in terms:
        end_if()
    def load_result():
        return res
    return load_result

def not_(term):
    return lambda: 1 - term()

def start_timer(timer_id=0):
    """ Start timer. Timer 0 runs from the start of the program. The
    total time of all used timers is output at the end. Fails if
    already running.

    :param timer_id: compile-time (int) """
    get_tape().start_new_basicblock(name='pre-start-timer')
    start(timer_id)
    get_tape().start_new_basicblock(name='post-start-timer')

def stop_timer(timer_id=0):
    """ Stop timer. Fails if not running.

    :param timer_id: compile-time (int) """
    get_tape().start_new_basicblock(name='pre-stop-timer')
    stop(timer_id)
    get_tape().start_new_basicblock(name='post-stop-timer')

def get_number_of_players():
    """
    :return: the number of players
    :rtype: regint
    """
    res = regint()
    nplayers(res)
    return res

def get_threshold():
    """ The threshold is the maximal number of corrupted
    players.

    :rtype: regint
    """
    res = regint()
    threshold(res)
    return res

def get_player_id():
    """
    :return: player number
    :rtype: localint (cannot be used for computation) """
    res = localint()
    playerid(res._v)
    return res

def listen_for_clients(port):
    """ Listen for clients on specific port base.

    :param port: port base (int/regint/cint)
    """
    instructions.listen(regint.conv(port))

def accept_client_connection(port):
    """ Accept client connection on specific port base.

    :param port: port base (int/regint/cint)
    :returns: client id
    """
    res = regint()
    instructions.acceptclientconnection(res, regint.conv(port))
    return res

def break_point(name=''):
    """
    Insert break point. This makes sure that all following code
    will be executed after preceding code.

    :param name: Name for identification (optional)
    """
    get_tape().start_new_basicblock(name=name)

def check_point():
    """
    Force MAC checks in current thread and all idle threads if the
    current thread is the main thread. This implies a break point.
    """
    break_point('pre-check')
    check()
    break_point('post-check')

# Fixed point ops

from math import ceil, log
from .floatingpoint import PreOR, TruncPr, two_power

def approximate_reciprocal(divisor, k, f, theta):
    """
        returns aproximation of 1/divisor
        where type(divisor) = cint
    """
    def twos_complement(x):
        bits = x.bit_decompose(k)[::-1]

        twos_result = cint(0)
        for i in range(k):
            val = twos_result
            val <<= 1
            val += 1 - bits[i]
            twos_result = val

        return twos_result + 1

    bits = divisor.bit_decompose(k)[::-1]

    flag = regint(0)
    cnt_leading_zeros = regint(0)
    normalized_divisor = divisor

    for i in range(k):
        flag = flag | (bits[i] == 1)
        flag_zero = cint(flag == 0)
        cnt_leading_zeros += flag_zero
        normalized_divisor <<= flag_zero

    q = two_power(k)
    e = twos_complement(normalized_divisor)

    for i in range(theta):
        q += (q * e) >> k
        e = (e * e) >> k

    res = q >> cint(2*k - 2*f - cnt_leading_zeros)

    return res


def cint_cint_division(a, b, k, f):
    """
        Goldschmidt method implemented with
        SE aproximation:
        http://stackoverflow.com/questions/2661541/picking-good-first-estimates-for-goldschmidt-division
    """
    # theta can be replaced with something smaller
    # for safety we assume that is the same theta from previous GS method

    if get_program().options.ring:
        assert 2 * f < int(get_program().options.ring)

    theta = int(ceil(log(k/3.5) / log(2)))
    two = cint(2) * two_power(f)

    sign_b = cint(1) - 2 * cint(b.less_than(0, k))
    sign_a = cint(1) - 2 * cint(a.less_than(0, k))
    absolute_b = b * sign_b
    absolute_a = a * sign_a
    w0 = approximate_reciprocal(absolute_b, k, f, theta)

    A = absolute_a
    B = absolute_b
    W = w0

    corr = cint(1) << (f - 1)

    for i in range(theta):
        A = (A * W + corr) >> f
        B = (B * W + corr) >> f
        W = two - B
    return (sign_a * sign_b) * A

from Compiler.program import Program
def sint_cint_division(a, b, k, f, kappa):
    """
        type(a) = sint, type(b) = cint
    """
    theta = int(ceil(log(k/3.5) / log(2)))
    two = cint(2) * two_power(f)
    sign_b = cint(1) - 2 * cint(b.less_than(0, k))
    sign_a = sint(1) - 2 * comparison.LessThanZero(a, k, kappa)
    absolute_b = b * sign_b
    absolute_a = a * sign_a
    w0 = approximate_reciprocal(absolute_b, k, f, theta)

    A = absolute_a
    B = absolute_b
    W = w0

    @for_range(1, theta)
    def block(i):
        A.link(TruncPr(A * W, 2*k, f, kappa))
        temp = (B * W) >> f
        W.link(two - temp)
        B.link(temp)
    return (sign_a * sign_b) * A

def IntDiv(a, b, k, kappa=None):
    l = 2 * k + 1
    return FPDiv(a.extend(l) << k, b.extend(l) << k, l, k,
                 kappa, nearest=True)

@instructions_base.ret_cisc
def FPDiv(a, b, k, f, kappa, simplex_flag=False, nearest=False):
    """
        Goldschmidt method as presented in Catrina10,
    """
    prime = get_program().prime
    if 2 * k == int(get_program().options.ring) or \
       (prime and 2 * k <= (prime.bit_length() - 1)):
        # not fitting otherwise
        nearest = True
    if get_program().options.binary:
        # no probabilistic truncation in binary circuits
        nearest = True
    res_f = f
    f = max((k - nearest) // 2 + 1, f)
    assert 2 * f > k - nearest
    theta = int(ceil(log(k/3.5) / log(2)))

    base.set_global_vector_size(b.size)
    alpha = b.get_type(2 * k).two_power(2*f, size=b.size)
    w = AppRcr(b, k, f, kappa, simplex_flag, nearest).extend(2 * k)
    x = alpha - b.extend(2 * k) * w
    base.reset_global_vector_size()

    l_y = k + 3 * f - res_f
    y = a.extend(l_y) * w
    y = y.round(l_y, f, kappa, nearest, signed=True)

    for i in range(theta - 1):
        x = x.extend(2 * k)
        y = y.extend(l_y) * (alpha + x).extend(l_y)
        x = x * x
        y = y.round(l_y, 2*f, kappa, nearest, signed=True)
        x = x.round(2*k, 2*f, kappa, nearest, signed=True)

    x = x.extend(2 * k)
    y = y.extend(l_y) * (alpha + x).extend(l_y)
    y = y.round(l_y, 3 * f - res_f, kappa, nearest, signed=True)
    return y

def AppRcr(b, k, f, kappa=None, simplex_flag=False, nearest=False):
    """
        Approximate reciprocal of [b]:
        Given [b], compute [1/b]
    """
    alpha = b.get_type(2 * k)(int(2.9142 * 2**k))
    c, v = b.Norm(k, f, kappa, simplex_flag)
    #v should be 2**{k - m} where m is the length of the bitwise repr of [b]
    d = alpha - 2 * c
    w = d * v
    w = w.round(2 * k + 1, 2 * (k - f), kappa, nearest, signed=True)
    # now w * 2 ^ {-f} should be an initial approximation of 1/b
    return w

def Norm(b, k, f, kappa, simplex_flag=False):
    """
        Computes secret integer values [c] and [v_prime] st.
        2^{k-1} <= c < 2^k and c = b*v_prime
    """
    # For simplex, we can get rid of computing abs(b)
    temp = None
    if simplex_flag == False:
        temp = comparison.LessThanZero(b, k, kappa)
    elif simplex_flag == True:
        temp = cint(0)

    sign = 1 - 2 * temp # 1 - 2 * [b < 0]
    absolute_val = sign * b

    #next 2 lines actually compute the SufOR for little indian encoding
    bits = absolute_val.bit_decompose(k, kappa, maybe_mixed=True)[::-1]
    suffixes = PreOR(bits, kappa)[::-1]

    z = [0] * k
    for i in range(k - 1):
        z[i] = suffixes[i] - suffixes[i+1]
    z[k - 1] = suffixes[k-1]

    acc = sint.bit_compose(reversed(z))

    part_reciprocal = absolute_val * acc
    signed_acc = sign * acc

    return part_reciprocal, signed_acc

def set_global_buildingblock(name):
    instructions.program.globalbuildingblock = name

def buildingblock(name):
    def decorator(func):
        def wrapper(*args, **kw):
            old_name = instructions.program.globalbuildingblock 
            set_global_buildingblock(name)
            get_tape().start_new_basicblock(name = name+"-start")
            res = func(*args, **kw)
            get_tape().start_new_basicblock(name = name + "-close")
            set_global_buildingblock(old_name)
        copy_doc(wrapper, func)
        return wrapper
    return decorator

def start_profiling():
    break_point()
    # instructions.program.is_profiling = True
    
def stop_profiling():
    break_point()
    # instructions.program.is_profiling = False


def mpc_psi_merge(*tables):
    from Compiler.sorting import gen_perm_by_radix_sort, SortPerm
    from Compiler.group_ops import GroupSum
    party_number = len(tables)
    num = 0
    attr = 1
    for table in tables:
        num = num + len(table)
        attr = attr + len(table[0]) - 1
    # merge all the table into one table
    final_table = tables[0].value_type.Matrix(num, attr)
    num_count = 0
    attr_count = 0

    for table in tables:
        table_attr = len(table[0])
        for ele in table:
            final_table[num_count][0] = ele[0]
            for i in range(1, table_attr):
                final_table[num_count][attr_count + i] = ele[i]
            num_count = num_count + 1
        attr_count = attr_count + table_attr - 1  # the id column should not be added

    ids = final_table.get_column(0)
    perm = gen_perm_by_radix_sort(ids)
    for i in range(attr):
        final_table.set_column(i, perm.apply(final_table.get_column(i)).get_vector())
    ids = final_table.get_column(0)
    flag = sint.Array(size=num)
    flag[0] = 1
    flag.assign_vector(ids.get_vector(size=len(ids) - 1) !=
                       ids.get_vector(size=len(ids) - 1, base=1), base=1)
    for i in range(1, attr):
        final_table.set_column(i, GroupSum(flag, final_table.get_column(i)))

    in_intersection = sint.Array(size=num)
    in_intersection.assign_vector(ids.get_vector(size=len(ids) - 1) ==
                       ids.get_vector(size=len(ids) - 1, base=party_number - 1), base=0)

    for i in range(attr):
        final_table.set_column(i, final_table.get_column(i) * in_intersection)
    perm = SortPerm(in_intersection.get_vector().bit_not())
    for i in range(attr):
        final_table.set_column(i, perm.apply(final_table.get_column(i)).get_vector())
    return final_table, sum(in_intersection)


