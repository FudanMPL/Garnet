/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Checks if the given list of associated Payload IR operations satisfy
/// the conditions defined by this type. If not, produces a silenceable
/// error at the specified location.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformTypeInterface::checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const {
      return getImpl()->checkPayload(getImpl(), *this, loc, payload);
  }
