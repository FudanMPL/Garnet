Homomorphic Encryption
----------------------

MP-SPDZ uses BGV encryption for triple generation in a number of
protocols. This involves zero-knowledge proofs in some protocols and
considerations about function privacy in all of them. The interface
described below allows directly accessing the basic cryptographic
operations in contexts where these considerations are not relevant.
See ``Utils/he-example.cpp`` for some example code.


Reference
~~~~~~~~~

.. doxygenclass:: FHE_Params
   :members:

.. doxygenclass:: FHE_KeyPair
   :members:

.. doxygenclass:: FHE_SK
   :members:

.. doxygenclass:: FHE_PK
   :members:

.. doxygenclass:: Plaintext
   :members:

.. doxygenclass:: Ciphertext
   :members:
