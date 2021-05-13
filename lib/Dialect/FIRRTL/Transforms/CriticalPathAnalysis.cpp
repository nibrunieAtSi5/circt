//===- CheckWidths.cpp - Check that width of types are known ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CheckWidths pass.
//
// Modeled after: <https://github.com/chipsalliance/firrtl/blob/master/src/main/
// scala/firrtl/passes/CheckWidths.scala>
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/Debug.h"

// todo: to be removed
#include <list>

#define DEBUG_TYPE "firrtl-critical-path"

using namespace circt;
using namespace firrtl;

namespace {
/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CriticalPathAnalysisPass : public CriticalPathAnalysisBase<CriticalPathAnalysisPass> {
  void runOnOperation() override;

  llvm::DenseMap<Operation*, double> opLatency;

  /// Whether any checks have failed.
  bool anyFailed;
};
} // namespace

void CriticalPathAnalysisPass::runOnOperation() {
  FModuleOp module = getOperation();
  anyFailed = false;

  LLVM_DEBUG(llvm::dbgs()
             << "executing CriticalPathAnalysisPass.\n");

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Port " << port.name << " of type " << port.type << "\n");
  }
  return;

  std::list<Operation*> worklist;
  double maxOpLatency = 0.0;
  Operation* criticalPathEnd = nullptr;

  // Check the results of each operation.
  module->walk([&](Operation *op) {
    opLatency[op] = -1.0;
    worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation* op = worklist.front();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing " << op->getName().getStringRef().str() << "\n");
    worklist.pop_front();
    auto it = opLatency.find(op);
    if (it == opLatency.end()) continue;

    double localLatency = it->second;
    if (localLatency >= 0.0) continue; // timing already evaluated
    // computing latency of operands
    bool allInputValid = true;
    double maxInpLatency = 0.0;
    // specialize per operation
    double localOpLatency = 1.0;
    for (auto inp : op->getOperands()) {
      if (localLatency < 0) {
        // one of the operand has no valid timing yet
        worklist.push_back(inp.getDefiningOp());
        allInputValid = false;
      }
      auto inp_it = opLatency.find(inp.getDefiningOp());
      if (inp_it == opLatency.end()) continue;
      if (inp_it->second > maxInpLatency) maxInpLatency = inp_it->second;
    }
    if (!allInputValid) worklist.push_back(op);
    else {
      localLatency = maxInpLatency + localOpLatency;
      opLatency[op] = localLatency;
      if (localLatency > maxOpLatency) {
        maxOpLatency = localLatency;
        criticalPathEnd = op;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs()
             << "critical path's end is " << criticalPathEnd->getName().getStringRef().str() << "\n");

  // looking for critical path

  markAllAnalysesPreserved();
}


std::unique_ptr<mlir::Pass> circt::firrtl::createCriticalPathAnalysisPass() {
  return std::make_unique<CriticalPathAnalysisPass>();
}

