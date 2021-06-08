//===- FIRRTLAnalyses.h - FIRRTL analysis passes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the FIRRTL analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLANALYSES_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLANALYSES_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
class Operation;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace firrtl {

/// A lookup table for fan-out count per operation
///
struct FanOutAnalysis {
  FanOutAnalysis(mlir::Operation *op);

  /// Return the fan-out count associed with the bitIndex-th bit of the resultIndex-th
  /// result of \p op
  int getFanOut(Operation *op, int bitIndex, int resultIndex=0) const;
private:
  /// Mapping from operations to the indexed fan-out count
  llvm::DenseMap<Operation *, llvm::SmallVector<int>> fanOutCount;

  /// evaluate the fan-out for a single bit signal of \p op
  int evaluateFanOut(Operation *op, int bitIndex, int resultIndex);

  /// evaluate the fan-out for a full operation \p op
  llvm::SmallVector<int> evaluateFanOut(Operation *op);

  void analyze(mlir::ModuleOp op);
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLANALYSES_H