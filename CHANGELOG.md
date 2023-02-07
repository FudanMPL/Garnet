The changelog explains changes pulled through from the private development repository. Bug fixes and small enhancements are committed between releases and not documented here.

## 0.3.4 (Nov 9, 2022)

- Decision tree learning
- Optimized oblivious shuffle in Rep3
- Optimized daBit generation in Rep3 and semi-honest HE-based 2PC
- Optimized element-vector AND in SemiBin
- Optimized input protocol in Shamir-based protocols
- Square-root ORAM (@Quitlox)
- Improved ORAM in binary circuits
- UTF-8 outputs

## 0.3.3 (Aug 25, 2022)

- Use SoftSpokenOT to avoid unclear security of KOS OT extension candidate
- Fix security bug in MAC check when using multithreading
- Fix security bug to prevent selective failure attack by checking earlier
- Fix security bug in Mama: insufficient sacrifice.
- Inverse permutation (@Quitlox)
- Easier direct compilation (@eriktaubeneck)
- Generally allow element-vector operations
- Increase maximum register size to 2^54
- Client example in Python
- Uniform base OTs across platforms
- Multithreaded base OT computation
- Faster random bit generation in two-player Semi(2k)

## 0.3.2 (May 27, 2022)

- Secure shuffling
- O(n log n) radix sorting
- Documented BGV encryption interface
- Optimized matrix multiplication in dealer protocol
- Fixed security bug in homomorphic encryption parameter generation
- Fixed security bug in Temi matrix multiplication

## 0.3.1 (Apr 19, 2022)

- Protocol in dealer model
- Command-line option for security parameter
- Fixed security bug in SPDZ2k (see Section 3.4 of [the updated paper](https://eprint.iacr.org/2018/482))
- Ability to run high-level (Python) code from C++
- More memory capacity due to 64-bit addressing
- Homomorphic encryption for more fields of characteristic two
- Docker container

## 0.3.0 (Feb 17, 2022)

- Semi-honest computation based on threshold semi-homomorphic encryption
- Batch normalization backward propagation
- AlexNet for CIFAR-10
- Specific private output protocols
- Semi-honest additive secret sharing without communication
- Sending of personal values
- Allow overwriting of persistence files
- Protocol signature in persistence files

## 0.2.9 (Jan 11, 2022)

- Disassembler
- Run-time parameter for probabilistic truncation error
- Probabilistic truncation for some protocols computing modulo a prime
- Simplified C++ interface
- Comparison as in [ACCO](https://dl.acm.org/doi/10.1145/3474123.3486757)
- More general scalar-vector multiplication
- Complete memory support for clear bits
- Extended clear bit functionality with Yao's garbled circuits
- Allow preprocessing information to be supplied via named pipes
- In-place operations for containers

## 0.2.8 (Nov 4, 2021)

- Tested on Apple laptop with ARM chip
- Restore trusted client interface
- Directly accessible softmax function
- Signature in preprocessing files to reduce confusing errors
- Improved error messages for connection issues
- Documentation of low-level share types and protocol pairs

## 0.2.7 (Sep 17, 2021)

- Optimized matrix multiplication in Hemi
- Improved client communication
- Private integer division as per [Veugen and Abspoel](https://doi.org/10.2478/popets-2021-0073)
- Compiler option to translate some Python control flow instructions
  to run-time instructions
- Functionality to break out of run-time loops
- Run-time range check of data structure accesses
- Improved documentation of network infrastructure

## 0.2.6 (Aug 6, 2021)

- [ATLAS](https://eprint.iacr.org/2021/833)
- Keras-like interface
- Iterative linear solution approximation
- Binary output
- HighGear/LowGear key generation for wider range of parameters by default
- Dabit generation for smaller primes and malicious security
- More consistent type model
- Improved local computation
- Optimized GF(2^8) for CCD
- NTL only needed for computation with GF(2^40)
- Virtual machines suggest compile-time optimizations
- Improved documentation of types

## 0.2.5 (Jul 2, 2021)

- Training of convolutional neural networks
- Bit decomposition using edaBits
- Ability to force MAC checks from high-level code
- Ability to close client connection from high-level code
- Binary operators for comparison results
- Faster compilation for emulation
- More documentation
- Fixed bug in dense layer back-propagation
- Fixed security bug: insufficient LowGear secret key randomness
- Fixed security bug: skewed random bit generation

## 0.2.4 (Apr 19, 2021)

- ARM support
- Base OTs optionally without SimpleOT/AVX
- Use OpenSSL instead of Crypto++ for elliptic curves
- Post-sacrifice binary computation with replicated secret sharing similar
  to [Araki et al.](https://www.ieee-security.org/TC/SP2017/papers/96.pdf)
- More flexible multithreading

## 0.2.3 (Feb 23, 2021)

- Distributed key generation for homomorphic encryption with active security similar to [Rotaru et al.](https://eprint.iacr.org/2019/1300)
- Homomorphic encryption parameters more similar to SCALE-MAMBA
- Fixed security bug: all-zero secret keys in homomorphic encryption
- Fixed security bug: missing check in binary Rep4
- Fixed security bug: insufficient "blaming" (covert security) in CowGear and ChaiGear due to low default security parameter

## 0.2.2 (Jan 21, 2021)

- Infrastructure for random element generation
- Programs generating as much preprocessing data as required by a particular high-level program
- Smaller binaries
- Cleaning up code
- Removing unused virtual machine instructions
- Fixed security bug: wrong MAC check in SPDZ2k input tuple generation

## 0.2.1 (Dec 11, 2020)

- Virtual machines automatically use the modulus used during compilation
- Non-linear computation modulo a prime without large gap in bit length
- Fewer communication rounds in several protocols

## 0.2.0 (Oct 28, 2020)

- Rep4: honest-majority four-party computation with malicious security
- SY/SPDZ-wise: honest-majority computation with malicious security based on replicated or Shamir secret sharing
- Training with a sequence of dense layers
- Training and inference for multi-class classification
- Local share conversion for semi-honest protocols based on additive secret sharing modulo a power of two
- edaBit generation based on local share conversion
- Optimize exponentiation with local share conversion
- Optimize Shamir pseudo-random secret sharing using a hyper-invertible matrix
- Mathematical functions (exponentiation, logarithm, square root, and trigonometric functions) with binary circuits
- Direct construction of fixed-point values from any type, breaking `sfix(x)` where `x` is the integer representation of a fixed-point number. Use `sfix._new(x)` instead.
- Optimized dot product for `sfix`
- Matrix multiplication via operator overloading uses VM-optimized multiplication.
- Fake preprocessing for daBits and edaBits
- Fixed security bug: insufficient randomness in SemiBin random bit generation.
- Fixed security bug: insufficient randomization of FKOS15 inputs.
- Fixed security bug in binary computation with SPDZ(2k).

## 0.1.9 (Aug 24, 2020)

- Streamline inputs to binary circuits
- Improved private output
- Emulator for arithmetic circuits
- Efficient dot product with Shamir's secret sharing
- Lower memory usage for TensorFlow inference
- This version breaks bytecode compatibility.

## 0.1.8 (June 15, 2020)

- Half-gate garbling
- Native 2D convolution
- Inference with some TensorFlow graphs
- MASCOT with several MACs to increase security

## 0.1.7 (May 8, 2020)

- Possibility of using global keyword in loops instead of MemValue
- IEEE754 floating-point functionality using Bristol Fashion circuits

## 0.1.6 (Apr 2, 2020)

- Bristol Fashion circuits
- Semi-honest computation with somewhat homomorphic encryption
- Use SSL for client connections
- Client facilities for all arithmetic protocols

## 0.1.5 (Mar 20, 2020)

- Faster conversion between arithmetic and binary secret sharing using [extended daBits](https://eprint.iacr.org/2020/338)
- Optimized daBits
- Optimized logistic regression
- Faster compilation of repetitive code (compiler option `-C`)
- ChaiGear: [HighGear](https://eprint.iacr.org/2017/1230) with covert key generation
- [TopGear](https://eprint.iacr.org/2019/035) zero-knowledge proofs
- Binary computation based on Shamir secret sharing
- Fixed security bug: Prove correctness of ciphertexts in input tuple generation
- Fixed security bug: Missing check in MASCOT bit generation and various binary computations

## 0.1.4 (Dec 23, 2019)

- Mixed circuit computation with secret sharing
- Binary computation for dishonest majority using secret sharing as in [FKOS15](https://eprint.iacr.org/2015/901)
- Fixed security bug: insufficient OT correlation check in SPDZ2k
- This version breaks bytecode compatibility.

## 0.1.3 (Nov 21, 2019)

- Python 3
- Semi-honest computation based on semi-homomorphic encryption
- Access to player information in high-level language

## 0.1.2 (Oct 11, 2019)

- Machine learning capabilities used for [MobileNets inference](https://eprint.iacr.org/2019/131) and the iDASH submission
- Binary computation for dishonest majority using secret sharing
- Mathematical functions from [SCALE-MAMBA](https://github.com/KULeuven-COSIC/SCALE-MAMBA)
- Fixed security bug: CowGear would reuse triples.

## 0.1.1 (Aug 6, 2019)

- ECDSA
- Loop unrolling with budget as in [HyCC](https://thomaschneider.de/papers/BDKKS18.pdf)
- Malicious replicated secret sharing for binary circuits
- New variants of malicious replicated secret over rings in [Use your Brain!](https://eprint.iacr.org/2019/164)
- MASCOT for any prime larger than 2^64
- Private fixed- and floating-point inputs

## 0.1.0 (Jun 7, 2019)

- CowGear protocol (LowGear with covert security)
- Protocols that sacrifice after than before
- More protocols for replicated secret sharing over rings
- Fixed security bug: Some protocols with supposed malicious security wouldn't check players' inputs when generating random bits.

## 0.0.9 (Apr 30, 2019)

- Complete BMR for all GF(2^n) protocols
- [Use your Brain!](https://eprint.iacr.org/2019/164)
- Semi/Semi2k for semi-honest OT-based computation
- Branching on revealed values in garbled circuits
- Fixed security bug: Potentially revealing too much information when opening linear combinations of private inputs in MASCOT and SPDZ2k with more than two parties

## 0.0.8 (Mar 28, 2019)

- SPDZ2k
- Integration of MASCOT and SPDZ2k preprocessing
- Integer division

## 0.0.7 (Feb 14, 2019)

- Simplified installation on macOS
- Optimized matrix multiplication
- Data type for quantization

## 0.0.6 (Jan 5, 2019)

- Shamir secret sharing

## 0.0.5 (Nov 5, 2018)

- More three-party replicated secret sharing
- Encrypted communication for replicated secret sharing

## 0.0.4 (Oct 11, 2018)

- Added BMR, Yao's garbled circuits, and semi-honest 3-party replicated secret sharing for arithmetic and binary circuits.
- Use inline assembly instead of MPIR for arithmetic modulo primes up length up to 128 bit.
- Added a secure multiplication instruction to the instruction set in order to accommodate protocols that don't use Beaver randomization.

## 0.0.3 (Mar 2, 2018)

- Added offline phases based on homomorphic encryption, used in the [SPDZ-2 paper](https://eprint.iacr.org/2012/642) and the [Overdrive paper](https://eprint.iacr.org/2017/1230).
- On macOS, the minimum requirement is now Sierra.
- Compilation with LLVM/clang is now possible (tested with 3.8).

## 0.0.2 (Sep 13, 2017)

### Support sockets based external client input and output to a SPDZ MPC program.

See the [ExternalIO directory](./ExternalIO/README.md) for more details and examples.

Note that [libsodium](https://download.libsodium.org/doc/) is now a dependency on the SPDZ build. 

Added compiler instructions:

* LISTEN
* ACCEPTCLIENTCONNECTION
* CONNECTIPV4
* WRITESOCKETSHARE
* WRITESOCKETINT

Removed instructions:

* OPENSOCKET
* CLOSESOCKET
 
Modified instructions:

* READSOCKETC
* READSOCKETS
* READSOCKETINT
* WRITESOCKETC
* WRITESOCKETS

Support secure external client input and output with new instructions:

* READCLIENTPUBLICKEY
* INITSECURESOCKET
* RESPSECURESOCKET

### Read/Write secret shares to disk to support persistence in a SPDZ MPC program.

Added compiler instructions:

* READFILESHARE
* WRITEFILESHARE

### Other instructions

Added compiler instructions:

* DIGESTC - Clear truncated hash computation
* PRINTINT - Print register value

## 0.0.1 (Sep 2, 2016)

### Initial Release

* See `README.md` and `tutorial.md`.
