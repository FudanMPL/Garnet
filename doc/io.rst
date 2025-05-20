.. _io:

Input/Output
------------

This section gives an overview over the input/output facilities.


Private Inputs from Computing Parties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All secret types have an input function
(e.g. :py:func:`Compiler.types.sint.get_input_from` or
:py:func:`Compiler.types.sfix.get_input_from`). Inputs are read as
whitespace-separated text in order (independent of the data type) from
``Player-Data/Input-P<player>-<thread>``, where ``thread`` is ``0`` for
the main thread. You can change the prefix (``Player-Data/Input``)
using the ``-IF`` option on the virtual machine binary. You can also
use ``-I`` to read inputs from the command line.
:py:func:`Compiler.types.sint.input_tensor_from` and
:py:func:`Compiler.types.sfix.input_tensor_from` allow inputting a
tensor.


Compile-Time Data via Private Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~Compiler.types.sint.input_tensor_via` is a convenience
function that allows to use data available at compile-time via
private input.


Public Inputs
~~~~~~~~~~~~~

All types can be assigned a hard-coded value at compile time, e.g.
``sint(1)``. This is impractical for larger amounts of
data. :py:func:`~Compiler.library.foreach_enumerate` provides a
facility for this case. It uses
:py:class:`~Compiler.library.public_input` internally, which reads
from ``Programs/Public-Input/<progname>``.


Public Outputs
~~~~~~~~~~~~~~

By default, :py:func:`~Compiler.library.print_ln` and related
functions only output to the terminal on party 0. This allows to run
several parties in one terminal without spoiling the output. You can
use interactive mode with option ``-I`` in order to output on all
parties. Note that this also to reading inputs from the command line
unless you specify ``-IF`` as well. You can also specify a file prefix
with ``-OF``, so that outputs are written to
``<prefix>-P<player>-<thread>``.


Private Outputs to Computing Parties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some types provide a function to reveal a value only to a specific
party (e.g., :py:func:`Compiler.types.sint.reveal_to`). It can be used
conjunction with :py:func:`~Compiler.library.print_ln_to` in order to
output it.


Binary Output
~~~~~~~~~~~~~

Most types returned by :py:func:`reveal` or :py:func:`reveal_to`
feature a :py:func:`binary_output` method, which writes to
``Player-Data/Binary-Output-P<playerno>-<threadno>``. The format is
either signed 64-bit integer or double-precision floating-point in
machine byte order (usually little endian).


Clients (Non-computing Parties)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`Compiler.types.sint.receive_from_client` and
:py:func:`Compiler.types.sint.reveal_to_clients` allow
communicating securely with the clients. See `this example
<https://github.com/data61/MP-SPDZ/tree/master/ExternalIO>`_
covering both client code and server-side high-level code.
:py:func:`Compiler.types.sint.input_tensor_from_client` and
:py:func:`Compiler.types.MultiArray.reveal_to_clients`. The same
functions are available for :py:class:`~Compiler.types.sfix` and
:py:class:`~Compiler.types.Array`, respectively.
See also :ref:`client ref` below.


Secret Shares
~~~~~~~~~~~~~

:py:func:`Compiler.types.sint.read_from_file` and
:py:func:`Compiler.types.sint.write_to_file` allow reading and writing
secret shares to and from files. These instructions use
``Persistence/Transactions-P<playerno>.data``. The format depends on
the protocol with the following principles.

- One share follows the other without metadata.
- If there is a MAC, it comes after the share.
- Numbers are stored in little-endian format.
- Numbers modulo a power of two are stored with the minimal number of
  bytes.
- Numbers modulo a prime are stored in Montgomery representation in
  blocks of eight bytes.

Another possibility for persistence between program runs is to use the
fact that the memory is stored in
``Player-Data/Memory-<protocol>-P<player>`` at the end of a run. The
best way to use this is via the memory access functions like
:py:func:`~Compiler.types.sint.store_in_mem` and
:py:func:`~Compiler.types.sint.load_mem`. Make sure to only use
addresses below ``USER_MEM`` specified in ``Compiler/config.py`` to
avoid conflicts with the automatic allocation used for arrays
etc. Note also that all types based on
:py:class:`~Compiler.types.sint` (e.g.,
:py:class:`~Compiler.types.sfix`) share the same memory, and that the
address is only a base address. This means that vectors will be
written to the memory starting at the given address.

.. _client ref:

Reference
~~~~~~~~~

.. doxygenclass:: Client
   :members:
