from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Generic, Type, TypeVar

from Compiler import library as lib
from Compiler import util
from Compiler.GC.types import cbit, sbit, sbitint, sbits
from Compiler.program import Program
from Compiler.types import (Array, MemValue, MultiArray, _clear, _secret, cint,
                            regint, sint, sintbit)
from Compiler.oram import demux_array, get_n_threads

# Adds messages on completion of heavy computation steps
debug = False
# Finer grained trace of steps that the ORAM performs
# + runtime error checks
# Warning: reveals information and makes the computation insecure
trace = False

n_threads = 16
n_parallel = 1024

# Avoids any memory allocation if set to False
# Setting to False prevents some optimizations but allows for controlling the ORAMs outside of the main tape
allow_memory_allocation = True


def get_n_threads(n_loops):
    if n_threads is None:
        if n_loops > 2048:
            return 8
        else:
            return None
    else:
        return n_threads


T = TypeVar("T", sint, sbitint)
B = TypeVar("B", sintbit, sbit)


class SqrtOram(Generic[T, B]):
    """Oblivious RAM using the "Square-Root" algorithm.

    :param MultiArray data: The data with which to initialize the ORAM. One may provide a MultiArray such that every "block" can hold multiple elements (an Array).
    :param sint value_type: The secret type to use, defaults to sint.
    :param int k: Leave at 0, this parameter is used to recursively pass down the depth of this ORAM.
    :param int period: Leave at None, this parameter is used to recursively pass down the top-level period.
    """
    # TODO: Preferably this is an Array of vectors, but this is currently not supported
    # One should regard these structures as Arrays where an entry may hold more
    # than one value (which is a nice property to have when using the ORAM in
    # practise).
    shuffle: MultiArray
    stash: MultiArray
    # A block has an index and data
    # `shuffle` and `stash` store the data,
    # `shufflei` and `stashi` store the index
    shufflei: Array
    stashi: Array

    shuffle_used: Array
    position_map: PositionMap

    # The size of the ORAM, i.e. how many elements it stores
    n: int
    # The period, i.e. how many calls can be made to the ORAM before it needs to be refreshed
    T: int
    # Keep track of how far we are in the period, and coincidentally how large
    # the stash is (each access results in a fake or real block being put on
    # the stash)
    t: cint

    def __init__(self, data: T | MultiArray, entry_length: int = 1, value_type: Type[T] = sint, k: int = 0, period: int | None = None, initialize: bool = True, empty_data=False) -> None:
        global debug, allow_memory_allocation

        # Correctly initialize the shuffle (memory) depending on the type of data
        if isinstance(data, MultiArray):
            self.shuffle = data
            self.n = len(data)
        elif isinstance(data, sint):
            self.n = math.ceil(len(data) // entry_length)
            if (len(data) % entry_length != 0):
                raise Exception('Data incorrectly padded.')
            self.shuffle = MultiArray(
                (self.n, entry_length), value_type=value_type)
            self.shuffle.assign_part_vector(data.get_vector())
        else:
            raise Exception("Incorrect format.")

        # Only sint is supported
        if value_type != sint and value_type != sbitint:
            raise Exception("The value_type must be either sint or sbitint")

        # Set derived constants
        self.value_type = value_type
        self.bit_type: Type[B] = value_type.bit_type
        self.index_size = util.log2(self.n) + 1 # +1 because signed
        self.index_type = value_type.get_type(self.index_size)
        self.entry_length = entry_length
        self.size = self.n

        if debug:
            lib.print_ln(
                'Initializing SqrtORAM of size %s at depth %s', self.n, k)

        self.shuffle_used = cint.Array(self.n)
        # Random permutation on the data
        self.shufflei = Array.create_from(
            [self.index_type(i) for i in range(self.n)])
        # Calculate the period if not given
        # upon recursion, the period should stay the same ("in sync"),
        # therefore it can be passed as a constructor parameter
        self.T = int(math.ceil(
            math.sqrt(self.n * util.log2(self.n) - self.n + 1))) if not period else period
        if debug and not period:
            lib.print_ln('Period set to %s', self.T)

        # Here we allocate the memory for the permutation
        # Note that self.shuffle_the_shuffle mutates this field
        #   Why don't we pass it as an argument then? Well, this way we don't have to allocate memory while shuffling, which keeps open the possibility for multithreading
        self.permutation = Array.create_from(
            [self.index_type(i) for i in range(self.n)])
        # We allow the caller to postpone the initialization of the shuffle
        # This is the most expensive operation, and can be done in a thread (only if you know what you're doing)
        # Note that if you do not initialize, the ORAM is insecure
        if initialize:
            # If the ORAM is not initialized with existing data, we can apply
            # a small optimization by forgoing shuffling the shuffle, as all
            # entries of the shuffle are equal and empty.
            if empty_data:
                random_shuffle = sint.get_secure_shuffle(self.n)
                self.shufflei.secure_permute(random_shuffle)
                self.permutation.assign(self.shufflei[:].inverse_permutation())
                if trace:
                    lib.print_ln('Calculated inverse permutation')
            else:
                self.shuffle_the_shuffle()
        else:
            print('You are opting out of default initialization for SqrtORAM. Be sure to call refresh before using the SqrtORAM, otherwise the ORAM is not secure.')
        # Initialize position map (recursive oram)
        self.position_map = PositionMap.create(self.permutation, k + 1, self.T)

        # Initialize stash
        self.stash = MultiArray((self.T, entry_length), value_type=value_type)
        self.stashi = Array(self.T, value_type=value_type)
        self.t = MemValue(cint(0))

        # Initialize temp variables needed during the computation
        self.found_ = self.bit_type.Array(size=self.T)
        self.j = MemValue(cint(0, size=1))

        # To prevent the compiler from recompiling the same code over and over again, we should use @method_block
        # However, @method_block requires allocation (of return address), which is not allowed when not in the main thread
        # Therefore, we only conditionally wrap the methods in a @method_block if we are guaranteed to be running in the main thread
        SqrtOram.shuffle_the_shuffle = lib.method_block(SqrtOram.shuffle_the_shuffle) if allow_memory_allocation else SqrtOram.shuffle_the_shuffle
        SqrtOram.refresh = lib.method_block(SqrtOram.refresh) if allow_memory_allocation else SqrtOram.refresh
        SqrtOram.reinitialize = lib.method_block(SqrtOram.reinitialize) if allow_memory_allocation else SqrtOram.reinitialize

    @lib.method_block
    def access(self, index: T, write: B, *value: T):
        global trace,n_parallel
        if trace:
            @lib.if_e(write.reveal() == 1)
            def _():
                lib.print_ln('Writing to secret index %s', index.reveal())

            @lib.else_
            def __():
                lib.print_ln('Reading from secret index %s', index.reveal())

        value = self.value_type(value, size=self.entry_length).get_vector(
            0, size=self.entry_length)
        index = MemValue(index)

        # Refresh if we have performed T (period) accesses
        @lib.if_(self.t == self.T)
        def _():
            self.refresh()

        found: B = MemValue(self.bit_type(False))
        result: T = MemValue(self.value_type(0, size=self.entry_length))

        # First we scan the stash for the item
        self.found_.assign_all(0)

        # This will result in a bit array with at most one True,
        # indicating where in the stash 'index' is found
        @lib.multithread(get_n_threads(self.T), self.T)
        def _(base, size):
            self.found_.assign_vector(
                (self.stashi.get_vector(base, size) == index.expand_to_vector(size)) &
                self.bit_type(regint.inc(size, base=base) <
                              self.t.expand_to_vector(size)),
                base=base)

        # To determine whether the item is found in the stash, we simply
        # check wheterh the demuxed array contains a True
        # TODO: What if the index=0?
        found.write(sum(self.found_))

        # Store the stash item into the result if found
        # If the item is not in the stash, the result will simple remain 0
        @lib.map_sum(get_n_threads(self.T), n_parallel, self.T,
                     self.entry_length, [self.value_type] * self.entry_length)
        def stash_item(i):
            entry = self.stash[i][:]
            access_here = self.found_[i]
            # This is a bit unfortunate
            # We should loop from 0 to self.t, but t is dynamic thus this is impossible.
            # Therefore we loop till self.T (the max value of self.t)
            # is_in_time = i < self.t

            # If we are writing, we need to add the value
            self.stash[i] += write * access_here * (value - entry)
            return (entry * access_here)[:]
        result += self.value_type(stash_item(), size=self.entry_length)

        if trace:
            @lib.if_e(found.reveal() == 1)
            def _():
                lib.print_ln('Found item in stash')

            @lib.else_
            def __():
                lib.print_ln('Did not find item in stash')

        # Possible fake lookup of the item in the shuffle,
        # depending on whether we already found the item in the stash
        physical_address = self.position_map.get_position(index, found)
        # We set shuffle_used to True, to track that this shuffle item needs to be refreshed
        # with its equivalent on the stash once the period is up.
        self.shuffle_used[physical_address] = cbit(True)

        # If the item was not found in the stash
        # ...we update the item in the shuffle
        self.shuffle[physical_address] += write * \
            found.bit_not() * (value - self.shuffle[physical_address][:])
        # ...and the item retrieved from the shuffle is our result
        result += self.shuffle[physical_address] * found.bit_not()
        # We append the newly retrieved item to the stash
        self.stash[self.t].assign(self.shuffle[physical_address][:])
        self.stashi[self.t] = self.shufflei[physical_address]

        if trace:
            @lib.if_((write * found.bit_not()).reveal())
            def _():
                lib.print_ln('Wrote (%s: %s) to shuffle[%s]', self.stashi[self.t].reveal(
                ), self.shuffle[physical_address].reveal(), physical_address)

        # Increase the "time" (i.e. access count in current period)
        self.t.iadd(1)

        return result

    @lib.method_block
    def write(self, index: T, *value: T):
        global trace, n_parallel
        if trace:
            lib.print_ln('Writing to secret index %s', index.reveal())

        if isinstance(value, tuple) or isinstance(value,list):
            value = self.value_type(value, size=self.entry_length)
            print(value, type(value))
        elif isinstance(value, self.value_type):
            value = self.value_type(*value, size=self.entry_length)
            print(value, type(value))
        else:
            raise Exception("Cannot handle type of value passed")
        print(self.entry_length, value, type(value),len(value))
        value = MemValue(value)
        index = MemValue(index)

        # Refresh if we have performed T (period) accesses
        @lib.if_(self.t == self.T)
        def _():
            self.refresh()

        found: B = MemValue(self.bit_type(False))
        result: T = MemValue(self.value_type(0, size=self.entry_length))

        # First we scan the stash for the item
        self.found_.assign_all(0)

        # This will result in an bit array with at most one True,
        # indicating where in the stash 'index' is found
        @lib.multithread(get_n_threads(self.T), self.T)
        def _(base, size):
            self.found_.assign_vector(
                (self.stashi.get_vector(base, size) == index.expand_to_vector(size)) &
                self.bit_type(regint.inc(size, base=base) <
                              self.t.expand_to_vector(size)),
                base=base)

        # To determine whether the item is found in the stash, we simply
        # check wheterh the demuxed array contains a True
        # TODO: What if the index=0?
        found.write(sum(self.found_))

        @lib.map_sum(get_n_threads(self.T), n_parallel, self.T,
                     self.entry_length, [self.value_type] * self.entry_length)
        def stash_item(i):
            entry = self.stash[i][:]
            access_here = self.found_[i]
            # This is a bit unfortunate
            # We should loop from 0 to self.t, but t is dynamic thus this is impossible.
            # Therefore we loop till self.T (the max value of self.t)
            # is_in_time = i < self.t

            # We update the stash value
            self.stash[i] += access_here * (value - entry)
            return (entry * access_here)[:]
        result += self.value_type(stash_item(), size=self.entry_length)

        if trace:
            @lib.if_e(found.reveal() == 1)
            def _():
                lib.print_ln('Found item in stash')

            @lib.else_
            def __():
                lib.print_ln('Did not find item in stash')

        # Possible fake lookup of the item in the shuffle,
        # depending on whether we already found the item in the stash
        physical_address = self.position_map.get_position(index, found)
        # We set shuffle_used to True, to track that this shuffle item needs to be refreshed
        # with its equivalent on the stash once the period is up.
        self.shuffle_used[physical_address] = cbit(True)

        # If the item was not found in the stash
        # ...we update the item in the shuffle
        self.shuffle[physical_address] += found.bit_not() * \
            (value - self.shuffle[physical_address][:])
        # ...and the item retrieved from the shuffle is our result
        result += self.shuffle[physical_address] * found.bit_not()
        # We append the newly retrieved item to the stash
        self.stash[self.t].assign(self.shuffle[physical_address][:])
        self.stashi[self.t] = self.shufflei[physical_address]

        if trace:
            @lib.if_(found.bit_not().reveal())
            def _():
                lib.print_ln('Wrote (%s: %s) to shuffle[%s]', self.stashi[self.t].reveal(
                ), self.shuffle[physical_address].reveal(), physical_address)

            lib.print_ln('Appended shuffle[%s]=(%s: %s) to stash at position t=%s', physical_address,
                         self.shufflei[physical_address].reveal(), self.shuffle[physical_address].reveal(), self.t)

        # Increase the "time" (i.e. access count in current period)
        self.t.iadd(1)

        return result

    @lib.method_block
    def read(self, index: T, *value: T):
        global debug, trace, n_parallel
        if trace:
            lib.print_ln('Reading from secret index %s', index.reveal())

        value = self.value_type(value)
        index = MemValue(index)

        # Refresh if we have performed T (period) accesses
        @lib.if_(self.t == self.T)
        def _():
            if debug:
                lib.print_ln('Refreshing SqrtORAM')
                lib.print_ln('t=%s according to me', self.t)

            self.refresh()

        found: B = MemValue(self.bit_type(False))
        result: T = MemValue(self.value_type(0, size=self.entry_length))

        # First we scan the stash for the item
        self.found_.assign_all(0)

        # This will result in a bit array with at most one True,
        # indicating where in the stash 'index' is found
        @lib.multithread(get_n_threads(self.T), self.T)
        def _(base, size):
            self.found_.assign_vector(
                (self.stashi.get_vector(base, size) == index.expand_to_vector(size)) &
                self.bit_type(regint.inc(size, base=base) <
                              self.t.expand_to_vector(size)),
                base=base)

        # To determine whether the item is found in the stash, we simply
        # check whether the demuxed array contains a True
        # TODO: What if the index=0?
        found.write(sum(self.found_))
        lib.check_point()

        # Store the stash item into the result if found
        # If the item is not in the stash, the result will simple remain 0
        @lib.map_sum(get_n_threads(self.T), n_parallel, self.T,
                     self.entry_length, [self.value_type] * self.entry_length)
        def stash_item(i):
            entry = self.stash[i][:]
            access_here = self.found_[i]
            # This is a bit unfortunate
            # We should loop from 0 to self.t, but t is dynamic thus this is impossible.
            # Therefore we loop till self.T (the max value of self.t)
            # is_in_time = i < self.t

            return (entry * access_here)[:]
        result += self.value_type(stash_item(), size=self.entry_length)

        if trace:
            # @lib.for_range(self.t)
            # def _(i):
            #     lib.print_ln("stash[%s]=(%s: %s)", i, self.stashi[i].reveal() ,self.stash[i].reveal())

            @lib.if_e(found.reveal() == 1)
            def _():
                lib.print_ln('Found item in stash (found=%s)', found.reveal())

            @lib.else_
            def __():
                lib.print_ln('Did not find item in stash (found=%s)', found.reveal())

        # Possible fake lookup of the item in the shuffle,
        # depending on whether we already found the item in the stash
        physical_address = self.position_map.get_position(index, found)
        # We set shuffle_used to True, to track that this shuffle item needs to be refreshed
        # with its equivalent on the stash once the period is up.
        self.shuffle_used[physical_address] = cbit(True)

        # If the item was not found in the stash
        # the item retrieved from the shuffle is our result
        result += self.shuffle[physical_address] * found.bit_not()
        # We append the newly retrieved item to the stash
        self.stash[self.t].assign(self.shuffle[physical_address][:])
        self.stashi[self.t] = self.shufflei[physical_address]

        if trace:
            lib.print_ln('Appended shuffle[%s]=(%s: %s) to stash at position t=%s', physical_address,
                         self.shufflei[physical_address].reveal(), self.shuffle[physical_address].reveal(), self.t)


        # Increase the "time" (i.e. access count in current period)
        self.t.iadd(1)

        return result

    __getitem__ = read
    __setitem__ = write

    def shuffle_the_shuffle(self) -> None:
        """Permute the memory using a newly generated permutation and return
        the permutation that would generate this particular shuffling.

        This permutation is needed to know how to map logical addresses to
        physical addresses, and is used as such by the postition map."""

        global trace
        # Random permutation on n elements
        random_shuffle = sint.get_secure_shuffle(self.n)
        if trace:
            lib.print_ln('Generated shuffle')
        # Apply the random permutation
        self.shuffle.secure_permute(random_shuffle)
        if trace:
            lib.print_ln('Shuffled shuffle')
        self.shufflei.secure_permute(random_shuffle)
        if trace:
            lib.print_ln('Shuffled shuffle indexes')

        lib.check_point()
        # Calculate the permutation that would have produced the newly produced
        # shuffle order. This can be calculated by regarding the logical
        # indexes (shufflei) as a permutation and calculating its inverse,
        # i.e. find P such that P([1,2,3,...]) = shufflei.
        # this is not necessarily equal to the inverse of the above generated
        # random_shuffle, as the shuffle may already be out of order (e.g. when
        # refreshing).
        self.permutation.assign(self.shufflei[:].inverse_permutation())
        # If shufflei does not contain exactly the indices 
        #       [i for i in range(self.n)], 
        # the underlying waksman network of 'inverse_permutation' will hang.
        if trace:
            lib.print_ln('Calculated inverse permutation')

    def refresh(self):
        """Refresh the ORAM by reinserting the stash back into the shuffle, and
        reshuffling the shuffle.

        This must happen on the T'th (period) accesses to the ORAM."""

        self.j.write(0)
        # Shuffle and emtpy the stash, and store elements back into shuffle

        @lib.for_range_opt(self.n)
        def _(i):
            @lib.if_(self.shuffle_used[i])
            def _():
                self.shuffle[i] = self.stash[self.j]
                self.shufflei[i] = self.stashi[self.j]
                self.j += 1

        # Reset the clock
        self.t.write(0)
        # Reset shuffle_used
        self._reset_shuffle_used()

        # Reinitialize position map
        self.shuffle_the_shuffle()
        # Note that we skip here the step of "packing" the permutation.
        # Since the underlying memory of the position map is already aligned in
        # this packed structure, we can simply overwrite the memory while
        # maintaining the structure.
        self.position_map.reinitialize(*self.permutation)

    def reinitialize(self, *data: T):
        # Note that this method is only used during refresh, and as such is
        # only called with a permutation as data.

        # The logical addresses of some previous permutation are irrelevant and must be reset
        self.shufflei.assign([self.index_type(i) for i in range(self.n)])
        # Reset the clock
        self.t.write(0)
        # Reset shuffle_used
        self._reset_shuffle_used()

        # Note that the self.shuffle is actually a MultiArray
        # This structure is preserved while overwriting the values using
        # assign_vector
        self.shuffle.assign_vector(self.value_type(
            data, size=self.n * self.entry_length))
        # Note that this updates self.permutation (see constructor for explanation)
        self.shuffle_the_shuffle()
        self.position_map.reinitialize(*self.permutation)

    def _reset_shuffle_used(self):
        global allow_memory_allocation
        if allow_memory_allocation:
            self.shuffle_used.assign_all(0)
        else:
            @lib.for_range_opt(self.n)
            def _(i):
                self.shuffle_used[i] = cint(0)


class PositionMap(Generic[T, B]):
    PACK_LOG: int = 3
    PACK: int = 1 << PACK_LOG

    n: int  # n in the paper
    depth: cint  # k in the paper
    value_type: Type[T]

    def __init__(self, n: int, value_type: Type[T] = sint, k: int = -1) -> None:
        self.n = n
        self.depth = MemValue(cint(k))
        self.value_type = value_type
        self.bit_type = value_type.bit_type
        self.index_type = self.value_type.get_type(util.log2(n) + 1) # +1 because signed

    @abstractmethod
    def get_position(self, logical_address: _secret, fake: B) -> Any:
        """Retrieve the block at the given (secret) logical address."""
        global trace
        if trace:
            print_at_depth(self.depth, 'Scanning %s for logical address %s (fake=%s)',
                         self.__class__.__name__, logical_address.reveal(), sintbit(fake).reveal())

    def reinitialize(self, *permutation: T):
        """Reinitialize this PositionMap.

        Since the reinitialization occurs at runtime (`on SqrtORAM.refresh()`),
        we cannot simply call __init__ on self. Instead, we must take care to
        reuse and overwrite the same memory.
        """
        ...

    @classmethod
    def create(cls, permutation: Array, k: int, period: int, value_type: Type[T] = sint) -> PositionMap:
        """Creates a new PositionMap. This is the method one should call when
        needing a new position map. Depending on the size of the given data, it
        will either instantiate a RecursivePositionMap or
        a LinearPositionMap."""
        n = len(permutation)

        global debug
        if n / PositionMap.PACK <= period:
            if debug:
                lib.print_ln(
                    'Initializing LinearPositionMap at depth %s of size %s', k, n)
            res = LinearPositionMap(permutation, value_type, k=k)
        else:
            if debug:
                lib.print_ln(
                    'Initializing RecursivePositionMap at depth %s of size %s', k, n)
            res = RecursivePositionMap(permutation, period, value_type, k=k)

        return res


class RecursivePositionMap(PositionMap[T, B], SqrtOram[T, B]):

    def __init__(self, permutation: Array, period: int, value_type: Type[T] = sint, k: int = -1) -> None:
        PositionMap.__init__(self, len(permutation), k=k)
        pack = PositionMap.PACK

        # We pack the permutation into a smaller structure, index with a new permutation
        packed_size = int(math.ceil(self.n / pack))
        packed_structure = MultiArray(
            (packed_size, pack), value_type=value_type)
        for i in range(packed_size):
            packed_structure[i] = Array.create_from(
                permutation[i*pack:(i+1)*pack])

        SqrtOram.__init__(self, packed_structure, value_type=value_type,
                          period=period, entry_length=pack, k=self.depth)

        # Initialize random temp variables needed during the computation
        self.block_index_demux: Array = self.bit_type.Array(self.T)
        self.element_index_demux: Array = self.bit_type.Array(PositionMap.PACK)

    @lib.method_block
    def get_position(self, logical_address: T, fake: B) -> _clear:
        super().get_position(logical_address, fake)

        pack = PositionMap.PACK
        pack_log = PositionMap.PACK_LOG

        # The item at logical_address
        # will be in block with index h (block.<h>)
        # at position l in block.data (block.data<l>)
        program = Program.prog
        h = MemValue(self.value_type.bit_compose(sbits.get_type(program.bit_length)(
            logical_address).right_shift(pack_log, program.bit_length)))
        l = self.value_type.bit_compose(sbits(logical_address) & (pack - 1))

        global trace
        if trace:
            print_at_depth(self.depth, '-> logical_address=%s:  h=%s, l=%s', logical_address.reveal(), h.reveal(), l.reveal())
            # @lib.for_range(self.t)
            # def _(i):
            #     print_at_depth(self.depth, "stash[%s]=(%s: %s)", i, self.stashi[i].reveal() ,self.stash[i].reveal())

        # The resulting physical address
        p = MemValue(self.index_type(-1))
        found: B = MemValue(self.bit_type(False))

        # First we try and retrieve the item from the stash at position stash[h][l]
        # Since h and l are secret, we do this by scanning the entire stash

        # First we scan the stash for the block we need
        self.block_index_demux.assign_all(0)
        @lib.for_range_opt_multithread(get_n_threads(self.T), self.T)
        def _(i):
            self.block_index_demux[i] = ( self.stashi[i] == h) & self.bit_type(i < self.t)
        # We can determine if the 'index' is in the stash by checking the
        # block_index_demux array
        found = sum(self.block_index_demux)
        # Once a block is found, we use the following condition to pick the correct item from that block
        demux_array(l.bit_decompose(PositionMap.PACK_LOG), self.element_index_demux)

        # Finally we use the conditions to conditionally write p
        @lib.map_sum(get_n_threads(self.T * pack), n_parallel, self.T * pack, 1, [self.value_type])
        def p_(i):
            # We should loop from 0 through self.t, but runtime loop lengths are not supported by map_sum
            # Therefore we include the check (i < self.t)
            return self.stash[i // pack][i % pack] * self.block_index_demux[i // pack] * self.element_index_demux[i % pack] * (i // pack< self.t)
        p.write(p_())

        if trace:
            @lib.if_e(found.reveal() == 0)
            def _(): print_at_depth(self.depth, 'Retrieve shuffle[%s]:', h.reveal())
            @lib.else_
            def __():
                print_at_depth(self.depth, 'Retrieve dummy element from shuffle:')

        # Then we try and retrieve the item from the shuffle (the actual memory)
        # Depending on whether we found the item in the stash, we either
        # block 'h' in which 'index' resides, or a random block from the shuffle
        p_prime = self.position_map.get_position(h, found)
        self.shuffle_used[p_prime] = cbit(True)

        # The block retrieved from the shuffle
        block_p_prime: Array = self.shuffle[p_prime]

        if trace:
            @lib.if_e(found.reveal() == 0)
            def _():
                print_at_depth(self.depth, 'Retrieved position from shuffle[%s]=(%s: %s)',
                             p_prime.reveal(), self.shufflei[p_prime].reveal(), self.shuffle[p_prime].reveal())

            @lib.else_
            def __():
                print_at_depth(self.depth, 'Retrieved dummy position from shuffle[%s]=(%s: %s)',
                             p_prime.reveal(), self.shufflei[p_prime].reveal(), self.shuffle[p_prime].reveal())

        # We add the retrieved block from the shuffle to the stash
        self.stash[self.t].assign(block_p_prime[:])
        self.stashi[self.t] = self.shufflei[p_prime]
        # Increase t
        self.t += 1

        # if found or not fake
        condition: B = self.bit_type(fake.bit_or(found.bit_not()))
        # Retrieve l'th item from block
        # l is secret, so we must use linear scan
        hit = Array.create_from((regint.inc(pack) == l.expand_to_vector(
            pack)) & condition.expand_to_vector(pack))

        @lib.for_range_opt(pack)
        def _(i):
            p.write((hit[i]).if_else(block_p_prime[i], p))

        return p.reveal()

    def reinitialize(self, *permutation: T):
        SqrtOram.reinitialize(self, *permutation)


class LinearPositionMap(PositionMap):
    physical: Array
    used: Array

    def __init__(self, data: Array, value_type: Type[T] = sint, k: int = -1) -> None:
        PositionMap.__init__(self, len(data), value_type, k=k)
        self.physical = data
        self.used = self.bit_type.Array(self.n)

        # Initialize random temp variables needed during the computation
        self.physical_demux: Array = self.bit_type.Array(self.n)

    @lib.method_block
    def get_position(self, logical_address: T, fake: B) -> _clear:
        """
        This method corresponds to GetPosBase in the paper.
        """
        super().get_position(logical_address, fake)

        global trace
        if trace:
            @lib.if_(((logical_address < 0) * (logical_address >= self.n)).reveal())
            def _():
                lib.runtime_error(
                    'logical_address must lie between 0 and self.n - 1')

        fake = MemValue(self.bit_type(fake))
        logical_address = MemValue(logical_address)

        p: MemValue = MemValue(self.index_type(-1))
        done: B = self.bit_type(False)

        # In order to get an address at secret logical_address,
        # we need to perform a linear scan.
        self.physical_demux.assign_all(0)

        @lib.for_range_opt_multithread(get_n_threads(self.n), self.n)
        def condition_i(i):
            self.physical_demux[i] = \
                (self.bit_type(fake).bit_not() & self.bit_type(logical_address == i)) \
                | (fake & self.used[i].bit_not())

        # In the event that fake=True, there are likely multiple entried in physical_demux set to True (i.e. where self.used[i] = False)
        # We only need once, so we pick the first one we find
        @lib.for_range_opt(self.n)
        def _(i):
            nonlocal done
            self.physical_demux[i] &= done.bit_not()
            done |= self.physical_demux[i]

        # Retrieve the value from the physical memory obliviously
        @lib.map_sum_opt(get_n_threads(self.n), self.n, [self.value_type])
        def calc_p(i):
            return self.physical[i] * self.physical_demux[i]
        p.write(calc_p())

        # Update self.used
        self.used.assign(self.used[:] | self.physical_demux[:])

        if trace:
            @lib.if_((p.reveal() < 0).bit_or(p.reveal() > len(self.physical)))
            def _():
                lib.runtime_error(
                    '%s Did not find requested logical_address in shuffle, something went wrong.', self.depth)

        return p.reveal()

    def reinitialize(self, *data: T):
        self.physical.assign_vector(data)

        global allow_memory_allocation
        if allow_memory_allocation:
            self.used.assign_all(False)
        else:
            @lib.for_range_opt(self.n)
            def _(i):
                self.used[i] = self.bit_type(0)

def print_at_depth(depth: cint, message: str, *kwargs):
    lib.print_str('%s', depth)
    @lib.for_range(depth)
    def _(i):
        lib.print_char(' ')
        lib.print_char(' ')
    lib.print_ln(message, *kwargs)
