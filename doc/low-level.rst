Low-Level Interface
===================

In the following we will explain the basic of the C++ interface by
walking trough :file:`Utils/paper-example.cpp`.

.. default-domain:: cpp

.. code-block:: cpp

    template<class T>
    void run(char** argv, int prime_length);

MP-SPDZ heavily uses templating to allow to reuse code between
different protocols. :cpp:func:`run` is a simple example of this.  The
entire virtual machine in the :file:`Processor` directory is built on
the same principle. The central type is a type representing a share in
a particular type.

.. code-block:: cpp

    // bit length of prime
    const int prime_length = 128;

    // compute number of 64-bit words needed
    const int n_limbs = (prime_length + 63) / 64;

Computation modulo a prime requires to fix the number of limbs (64-bit
words) at compile time. This allows for optimal memory usage and
computation.

.. code-block:: cpp

    if (protocol == "MASCOT")
        run<Share<gfp_<0, n_limbs>>>(argv, prime_length);
    else if (protocol == "CowGear")
        run<CowGearShare<gfp_<0, n_limbs>>>(argv, prime_length);

Share types for computation module a prime (and in
:math:`\mathrm{GF}(2^n)`) generally take one parameter for the
computation domain. :class:`gfp_` in turn takes two parameters, a
counter and the number of limbs. The counter allows to use several
instances with different parameters. It can be chosen freely, but the
convention is to use 0 for the online phase and 1 for the offline
phase where required.

.. code-block:: cpp

    else if (protocol == "SPDZ2k")
        run<Spdz2kShare<64, 64>>(argv, 0);

Share types for computation modulo a power of two simply take the
exponent as parameter, and some take an additional security parameter.

.. code-block:: cpp

    int my_number = atoi(argv[1]);
    int n_parties = atoi(argv[2]);
    int port_base = 9999;
    Names N(my_number, n_parties, "localhost", port_base);

All implemented protocols require point-to-point connections between
all parties. :class:`Names` objects represent a setup of hostnames and
IPs used to set up the actual
connections. The chosen initialization provides a way where
every party connects to party 0 on a specified location (localhost in
this case), which then broadcasts the locations of all parties. The
base port number is used to derive the port numbers for the parties to
listen on (base + party number). See the the :class:`Names` class for
other possibilities such as a text file containing hostname and port
number for each party.

.. code-block:: cpp

    CryptoPlayer P(N);

The networking setup is used to set up the actual
connections. :class:`CryptoPlayer` uses encrypted connection while
:class:`PlainPlayer` does not. If you use several instances (for
several threads for example), you must use an integer identifier as
the second parameter, which must differ from any other by at least the
number of parties.

.. code-block:: cpp

    ProtocolSetup<T> setup(P, prime_length);

We have to use a specific prime for computation modulo a prime. This
deterministically generates one of the desired length if
necessary. For computation modulo a power of two, this does not do
anything.  Some protocols use an information-theoretic tag that is
constant throughout the protocol. This code reads it from storage if
available or generates a fresh one otherwise.

.. code-block:: cpp

    ProtocolSet<T> set(P, setup);
    auto& input = set.input;
    auto& protocol = set.protocol;
    auto& output = set.output;

The :class:`ProtocolSet<T>` contains one instance for every essential
protocol step.

.. code-block:: cpp

    int n = 1000;
    vector<T> a(n), b(n);
    T c;
    typename T::clear result;

Remember that :type:`T` stands for a share in the protocol. The
derived type :type:`T::clear` stands for the cleartext domain. Share
types support linear operations such as addition, subtraction, and
multiplication with a constant. Use :func:`T::constant` to convert a
constant to a share type.

.. code-block:: cpp

    input.reset_all(P);
    for (int i = 0; i < n; i++)
        input.add_from_all(i);
    input.exchange();
    for (int i = 0; i < n; i++)
    {
        a[i] = input.finalize(0);
        b[i] = input.finalize(1);
    }

The interface for all protocols proceeds in four stages:

1. Initialization. This is required to initialize and reset data
   structures in consecutive use.
2. Local data preparation
3. Communication
4. Output extraction

This blueprint allows for a minimal number of communication rounds.

.. code-block:: cpp

    protocol.init_dotprod(&processor);
    for (int i = 0; i < n; i++)
        protocol.prepare_dotprod(a[i], b[i]);
    protocol.next_dotprod();
    protocol.exchange();
    c = protocol.finalize_dotprod(n);

The initialization of the multiplication sets the preprocessing and
output instances to use in Beaver multiplication. :func:`next_dotprod`
separates dot products in the data preparation phase.

.. code-block:: cpp

    set.check();

Some protocols require a check of all multiplications up to a certain
point. To guarantee that outputs do not reveal secret information, it
has to be run before using the output protocol.

.. code-block:: cpp

    output.init_open(P);
    output.prepare_open(c);
    output.exchange(P);
    result = output.finalize_open();

    cout << "result: " << result << endl;
    output.Check(P);

The output protocol follows the same blueprint as the multiplication
protocol.

.. code-block:: cpp

    set.check();

Some output protocols require an additional to guarantee the
correctness of outputs.


Thread Safety
-------------

The low-level interface generally isn't thread-safe. In particular,
you should only use one instance of :class:`ProtocolSetup` in the
whole program, and you should use only one instance of
:class:`CryptoPlayer`/:class:`PlainPlayer` and :class:`ProtocolSet`
per thread.


Domain Types
------------

.. list-table::
   :widths: 20 80

   *
     - ``gfp_<X, L>``
     - Computation modulo a prime. ``L`` is the number of 64-bit
       limbs, that is, it covers primes of bit length
       :math:`64(L-1)+1` to :math:`64L`. The type has to be
       initialized using :cpp:func:`init_field` or
       :cpp:func:`init_default`. The latter picks a prime given a bit length.
   *
     - ``SignedZ2<K>`` / ``Z2<K>``
     - Computation modulo :math:`2^K`. This is not a field.
   *
     - ``gf2n_short`` / ``gf2n_long`` / ``gf2n_<T>``
     - :math:`GF(2^n)`. ``T`` denotes a type that is used to store the
       values. It must support a variety of integer
       operations. The type has to be initialized using
       :cpp:func:`init_field`. The choice of degrees is limited. At
       the time of writing, 4, 8, 28, 40, 63, and 128 are supported if the
       storage type is large enough.


.. _share-type-reference:

Share Types
------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   *
     - Type
     - Protocol
   *
     - ``AtlasShare<T>``
     - Semi-honest version of `ATLAS
       <https://eprint.iacr.org/2021/833>`_ (Section 4.2). ``T`` must
       represent a field.
   *
     - ``ChaiGearShare<T>``
     - `HighGear <https://eprint.iacr.org/2017/1230>`_ with covert key
       setup. ``T`` must be ``gfp_<X, L>`` or ``gf2n_short``.
   *
     - ``CowGearShare<T>``
     - `LowGear <https://eprint.iacr.org/2017/1230>`_ with covert key
       setup. ``T`` must be ``gfp_<X, L>`` or ``gf2n_short``.
   *
     - ``HemiShare<T>``
     - Semi-honest protocol with Beaver multiplication based on
       semi-homomorphic encryption. ``T`` must be ``gfp_<X, L>`` or
       ``gf2n_short``.
   *
     - ``HighGearShare<T>``
     - `HighGear <https://eprint.iacr.org/2017/1230>`_. ``T`` must be
       ``gfp_<X, L>`` or ``gf2n_short``.
   *
     - ``LowGearShare<T>``
     - `LowGear <https://eprint.iacr.org/2017/1230>`_. ``T`` must be
       ``gfp_<X, L>`` or ``gf2n_short``.
   *
     - ``MaliciousShamirShare<T>``
     - Shamir secret sharing with Beaver multiplication and sacrifice.
       ``T`` must represent a field.
   *
     - ``MamaShare<T, N>``
     - `MASCOT <https://eprint.iacr.org/2016/505>`_ with multiple
       MACs. ``T`` must represent a field, ``N`` is the number of MACs.
   *
     - ``PostSacriRepFieldShare<T>``
     - `Post-sacrifice <https://eprint.iacr.org/2017/816>`_ protocol
       using three-party replicated secret sharing with ``T``
       representing a field.
   *
     - ``PostSacriRepRingShare<K, S>``
     - `Post-sacrifice protocol <https://eprint.iacr.org/2019/164>`_
       using replicated three-party secret sharing modulo :math:`2^K`
       with security parameter ``S``.
   *
     - ``Rep3Share2<K>``
     - `Three-party semi-honest protocol
       <https://eprint.iacr.org/2016/768>`_ using replicated secret
       sharing modulo :math:`2^K`.
   *
     - ``Rep4Share<T>``
     - `Four-party malicious protocol
       <https://eprint.iacr.org/2020/1330>`_ using replicated secret
       sharing over a field.
   *
     - ``Rep4Share2<K>``
     - `Four-party malicious protocol
       <https://eprint.iacr.org/2020/1330>`_ using replicated secret
       sharing modulo :math:`2^K`.
   *
     - ``SemiShare2<K>``
     - Semi-honest dishonest-majority protocol using Beaver
       multiplication based on oblivious transfer modulo :math:`2^K`.
   *
     - ``SemiShare<T>``
     - Semi-honest dishonest-majority protocol using Beaver
       multiplication based on oblivious transfer in a field.
   *
     - ``ShamirShare<T>``
     - `Semi-honest protocol <https://eprint.iacr.org/2000/037>`_
       based on Shamir's secret sharing. ``T`` must represent a field.
   *
     - ``Share<T>``
     - `MASCOT <https://eprint.iacr.org/2016/505>`_. ``T`` must
       represent a field.
   *
     - ``SohoShare<T>``
     - Semi-honest protocol with Beaver multiplication based on
       somewhat homomorphic encryption. ``T`` must be ``gfp_<X, L>``
       or ``gf2n_short``.
   *
     - ``Spdz2kShare<K, S>``
     - `SPDZ2k <https://eprint.iacr.org/2018/482>`_ computing modulo
       :math:`2^K` with security parameter ``S``.
   *
     - ``SpdzWiseShare<K, S>``
     - `SPDZ-wise <https://eprint.iacr.org/2019/1298>`_ computing
       modulo :math:`2^K` with security parameter ``S``.
   *
     - ``SpdzWiseShare<T>``
     - `SPDZ-wise <https://eprint.iacr.org/2018/570>`_. ``T`` must be
       ``MaliciousShamirShare`` or ``MaliciousRep3Share``.
   *
     - ``TemiShare<T>``
     - Semi-honest protocol with Beaver multiplication based on
       threshold semi-homomorphic encryption. ``T`` must be
       ``gfp_<X, L>`` or ``gf2n_short``.


Protocol Setup
--------------

.. doxygenclass:: ProtocolSetup
   :members:

.. doxygenclass:: ProtocolSet
   :members:

.. doxygenclass:: BinaryProtocolSetup
   :members:

.. doxygenclass:: BinaryProtocolSet
   :members:

.. doxygenclass:: MixedProtocolSetup
   :members:

.. doxygenclass:: MixedProtocolSet
   :members:


Protocol Interfaces
-------------------

.. doxygenclass:: ProtocolBase
   :members:

.. doxygenclass:: InputBase
   :members:

.. doxygenclass:: MAC_Check_Base
   :members:

.. doxygenclass:: Preprocessing
   :members:

.. doxygenclass:: BufferPrep
   :members:


Domain Reference
----------------

.. doxygenclass:: gfp_
   :members:

.. doxygenclass:: gfpvar_
   :members:

.. doxygenclass:: Z2
   :members:

.. doxygenclass:: SignedZ2
   :members:
