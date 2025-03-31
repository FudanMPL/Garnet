/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns true if the operation may have a passthru argument when masked.
bool mlir::vector::MaskableOpInterface::supportsPassthru() {
      return getImpl()->supportsPassthru(getImpl(), getOperation());
  }
/// Returns the mask type expected by this operation. It requires the operation to be vectorized.
mlir::VectorType mlir::vector::MaskableOpInterface::getExpectedMaskType() {
      return getImpl()->getExpectedMaskType(getImpl(), getOperation());
  }
