.. _nonlinear:

Non-linear Computation
----------------------

While the computation of addition and multiplication varies from
protocol, non-linear computation such as comparison in arithmetic
domains (modulus other than two) only comes in three flavors
throughout MP-SPDZ:

Unknown prime modulus
    This approach goes back to `Catrina and de Hoogh
    <https://www.researchgate.net/profile/Sebastiaan-Hoogh/publication/225092133_Improved_Primitives_for_Secure_Multiparty_Integer_Computation/links/0c960533585ad99868000000/Improved-Primitives-for-Secure-Multiparty-Integer-Computation.pdf>`_. It crucially relies on
    the use of secret random bits in the arithmetic domain. Enough
    such bits allow to mask a secret value so that it is secure to
    reveal the masked value. This can then be split in bits as it is
    public. The public bits and the secret mask bits are then used to
    compute a number of non-linear functions. The same idea has been
    used to implement `fixed-point
    <https://www.ifca.ai/pub/fc10/31_47.pdf>`_ and
    `floating-point <https://eprint.iacr.org/2012/405>`_ computation.
    We call this method "unknown prime modulus" because it only
    mandates a minimum modulus size for a given cleartext range, which
    is roughly the cleartext bit length plus a statistical security
    parameter. It has the downside that there is implicit enforcement
    of the cleartext range.

Known prime modulus
    `Damgård et al. <https://doi.org/10.1007/11681878_15>`_ have
    proposed non-linear computation that involves an exact prime
    modulus. We have implemented the refined bit decomposition by
    `Nishide and Ohta
    <https://doi.org/10.1007/978-3-540-71677-8_23>`_, which enables
    further non-linear computation. Our assumption with this method is
    that the cleartext space is slightly smaller the full range modulo
    the prime. This allows for comparison by taking a difference and
    extracting the most significant bit, which is different than the
    above works that implement comparison between two positive numbers
    modulo the prime. We also used an idea by `Makri et
    al. <https://eprint.iacr.org/2021/119>`_, namely that a random
    :math:`k`-bit number is indistinguishable from a random number
    modulo :math:`p` if the latter is close enough to :math:`2^k`.

Power-of-two modulus
    In the context of non-linear computation, there are two important
    differences to prime modulus setting:

    1. Multiplication with a power of two effectively erases some of
       the most significant bits.

    2. There is no right shift using multiplication. Modulo a prime,
       multiplying with a power of the inverse of two allows to
       right-shift numbers with enough zeros as least significant
       bits.

    Taking this differences into account, `Dalskov et
    al. <https://eprint.iacr.org/2019/131>`_ have adapted the
    mask-and-reveal approach above to the setting of computation
    modulo a power of two.

See also `this slide deck
<https://mkskeller.github.io/files/programming.pdf>`_ for an
introduction to non-linear computation in arithmetic MPC.


Mixed-Circuit Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

Another approach to non-linear computation is switching to binary
computation for parts of the computation. MP-SPDZ implements protocols
proposed for particular security models by a number of works: `Demmler et
al. <https://encrypto.de/papers/DSZ15.pdf>`_, `Mohassel and Rindal
<https://eprint.iacr.org/2018/403>`_, and `Dalskov et
al. <https://eprint.iacr.org/2020/1330>`_ MP-SPDZ also implements
more general methods such as `daBits
<https://eprint.iacr.org/2019/207>`_ and `edaBits
<https://eprint.iacr.org/2020/338>`_.

See also `this slide deck
<https://mkskeller.github.io/files/mixed.pdf>`_ for an introduction to
mixed-circuit computation.


.. _protocol-pairs:

Protocol Pairs
==============

The following table lists the matching arithmetic and binary protocols.

.. list-table::
   :header-rows: 1

   *
     - Arithmetic
     - Binary
   *
     - MASCOT, SPDZ2k, LowGear, HighGear, CowGear, ChaiGear
     - `Tinier <https://eprint.iacr.org/2015/901>`_ with improved
       cut-and-choose analysis by `Furukawa et
       al. <https://eprint.iacr.org/2016/944>`_
   *
     - Semi, Hemi, Temi, Soho, Semi2k
     - SemiBin (Beaver triples modulo 2 using OT)
   *
     - `Malicious Shamir <https://eprint.iacr.org/2017/816>`_
     - Malicious Shamir over :math:`GF(2^{40})` for secure sacrificing
   *
     - Malicious Rep3, Post-Sacrifice, SPDZ-wise replicated
     - `Malicious Rep3 modulo 2 <https://eprint.iacr.org/2016/944>`_
   *
     - `Rep4 <https://eprint.iacr.org/2020/1330>`_
     - Rep4 modulo 2
   *
     - `Shamir <https://eprint.iacr.org/2000/037>`_
     - Shamir over :math:`GF(2^8)`
   *
     - `ATLAS <https://eprint.iacr.org/2021/833>`_
     - ATLAS over :math:`GF(2^8)`
   *
     - `Rep3 <https://eprint.iacr.org/2016/768>`_
     - Rep3
