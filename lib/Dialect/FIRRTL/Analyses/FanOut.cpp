// Quick & dirty determination of operation fan-out
//===- LegalNames.cpp - SV/RTL name legalization analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis pass establishes legalized names for SV/RTL operations that are
// safe to use in SV output.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnalyses.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Fan-Out Lookup
//===----------------------------------------------------------------------===//

int FanOutAnalysis::getFanOut(Operation *op, int bitIndex, int resultIndex) const {
  auto nameIt = fanOutCount.find(op);
  assert(nameIt != fanOutCount.end() && "expected known operation");
  return nameIt->second[bitIndex];
}

//===----------------------------------------------------------------------===//
// Fan-Out Evaluation
//===----------------------------------------------------------------------===//

int FanOutAnalysis::evaluateFanOut(Operation *op, int bitIndex, int resultIndex) {
  // a single entry will store the fan-out for every bit of <op>
  auto &entry = fanOutCount[op];
  if (entry.empty())
    entry = evaluateFanOut(op);

  return entry[bitIndex];
}

llvm::SmallVector<int> FanOutAnalysis::evaluateFanOut(Operation *op) {
  llvm::SmallVector<int> result;
  return result;
}

