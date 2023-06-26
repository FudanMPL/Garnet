Networking
----------

All protocols in MP-SPDZ rely on point-to-point connections between
all pairs of parties. This is realized using TCP, which means that
every party must be reachable under at least one TCP port. The default
is to set this port to a base plus the player number. This allows for
easily running all parties on the same host. The base defaults to 5000,
which can be changed with the command-line option
``--portnumbase``. However, the scripts in ``Scripts`` use a random
base port number, which can be changed using the same option.

There are two ways of communicating hosts and
individually setting ports:

1. All parties first connect to a coordination server, which
   broadcasts the data for all parties. This is the default with the
   coordination server being run as a thread of party 0. The hostname
   of the coordination server has to be given with the command-line
   parameter ``--hostname``, and the coordination server runs on the
   base port number, thus defaulting to 5000. Furthermore, you
   can specify a party's listening port using ``--my-port``.

2. The parties read the information from a local file, which needs to
   be the same everywhere. The file can be specified using
   ``--ip-file-name`` and has the following format::

     <host0>[:<port0>]
     <host1>[:<port1>]
     ...

   The hosts can be both hostnames and IP addresses. If not given, the
   ports default to base plus party number.

Whether or not encrypted connections are used depends on the security
model of the protocol. Honest-majority protocols default to encrypted
whereas dishonest-majority protocols default to unencrypted. You
change this by either using ``--encrypted/-e`` or
``--unencrypted/-u``.

If using encryption, the certificates (``Player-Data/*.pem``) must be
the same on all hosts, and you have to run ``c_rehash Player-Data`` on
all of them. ``Scripts/setup-ssl.sh`` can be used to generate the
necessary certificates. The common name has to be ``P<player number>``
for computing parties and ``C<client number>`` for clients.


.. _network-reference:

Internal Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~

The internal networking infrastructure of MP-SPDZ reflects the needs
of the various multi-party computation. For example, some protocols
require a simultaneous broadcast from all parties whereas other
protocols require that every party sends different information to
different parties (include none at all). The infrastructure makes sure
to send and receive in parallel whenever possible.

All communication is handled through two subclasses of :cpp:class:`Player`
defined in ``Networking/Player.h``. :cpp:class:`PlainPlayer` communicates
in cleartext while :cpp:class:`CryptoPlayer` uses TLS encryption. The
former uses the same BSD socket for sending and receiving but the
latter uses two different connections for sending and receiving. This
is because TLS communication is never truly one-way due key renewals
etc., so the only way for simultaneous sending and receiving we found
was to use two connections in two threads.

If you wish to use a different networking facility, we recommend to
subclass :cpp:class:`Player` and fill in the virtual-only functions
required by the compiler (e.g., :cpp:func:`send_to_no_stats` for
sending to one other party). Note that not all protocols require all
functions, so you only need to properly implement those you need. You
can then replace uses of :cpp:class:`PlainPlayer` or
:cpp:class:`CryptoPlayer` by your own class. Furthermore, you might
need to extend the :class:`Names` class to suit your purpose. By
default, :cpp:class:`Names` manages one TCP port that a party is
listening on for connections. If this suits you, you don't need to
change anything


Reference
=========

.. doxygenclass:: Names
   :members:
   :undoc-members:

.. doxygenclass:: Player
   :members:

.. doxygenclass:: MultiPlayer
   :members:

.. doxygenclass:: PlainPlayer
   :members:

.. doxygenclass:: CryptoPlayer
   :members:

.. doxygenclass:: octetStream
   :members:
