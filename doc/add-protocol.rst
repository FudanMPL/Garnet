Adding a Protocol
-----------------

In order to illustrate how to create a virtual machine for a new
protocol, we have created one with blanks to be filled in. It is
defined in the following files:

``Machines/no-party.cpp``
  Contains the main function.

``Protocols/NoShare.h``
  Contains the :c:type:`NoShare` class, which is supposed to hold one
  share. :c:type:`NoShare` takes the cleartext type as a template
  parameter.

``Protocols/NoProtocol.h``
  Contains a number of classes representing instances of protocols:

    :c:type:`NoInput`
      Private input.
    :c:type:`NoProtocol`
      Multiplication protocol.
    :c:type:`NoOutput`
      Public output.

``Protocols/NoLivePrep.h``
  Contains the :c:type:`NoLivePrep` class, representing a
  preprocessing instance.

The number of blanks can be overwhelming. We therefore recommend the
following approach to get started. If the desired protocol resembles
one that is already implemented, you can check its code for
inspiration. The main function of ``<protocol>-party.x`` can be found
in ``Machines/<protocol>-party.cpp``, which in turns contains the name
of the share class. For example ``replicated-ring-party.x`` is
implemented in ``Machines/replicated-ring-party.cpp``, which refers to
:c:func:`Rep3Share2` in ``Protocols/Rep3Share2.h``. There you will
find that it uses :c:func:`Replicated` for multiplication, which is
found in ``Protocols/Replicated.h``.

1. Fill in the :c:func:`constant` static member function of
   :c:type:`NoShare` as well as the :c:func:`exchange` member function
   of :c:type:`NoOutput`. Check out
   :c:func:`DirectSemiMC<T>::exchange_` in ``Protocols/SemiMC.hpp``
   for a simple example. It opens an additive secret sharing by
   sending all shares to all other parties and then summing up the
   received. See :ref:`this reference <network-reference>` for
   documentation on the necessary infrastructure.
   Constant sharing and public output allows to execute the
   following program::

     print_ln('%s', sint(123).reveal())

   This allows to check the correct execution of further
   functionality.

2. Fill in the operator functions in :c:type:`NoShare` and check
   them::

     print_ln('%s', (sint(2) + sint(3)).reveal())
     print_ln('%s', (sint(2) - sint(3)).reveal())
     print_ln('%s', (sint(2) * cint(3)).reveal())

   Many protocols use these basic operations, which makes it
   beneficial to check the correctness

3. Fill in :c:type:`NoProtocol`. Alternatively, if the desired
   protocol is based on Beaver multiplication, you can specify the
   following in :c:type:`NoShare`::

     typedef Beaver<This> Protocol;

   Then add the desired triple generation to
   :c:func:`NoLivePrep::buffer_triples()`. In
   any case you should then be able to execute::

     print_ln('%s', (sint(2) * sint(3)).reveal())

4. In order to execute many kinds of non-linear computation, random
   bits are needed. After filling in
   :c:func:`NoLivePrep::buffer_bits()`, you should be able to
   execute::

     print_ln('%s', (sint(2) < sint(3)).reveal()


Reference
=========

The following classes are fundamental building blocks in
protocols. See also the :ref:`this reference <network-reference>`
for networking-related classes.

.. doxygenclass:: PRNG
   :members:

.. doxygenclass:: SeededPRNG
   :members:

.. doxygenclass:: GlobalPRNG
   :members:
