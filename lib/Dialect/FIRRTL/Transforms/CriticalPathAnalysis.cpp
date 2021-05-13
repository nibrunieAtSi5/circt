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

  // llvm::DenseMap<Operation, double> opsLatency;
  llvm::DenseMap<Value, double> valuesLatency;

  /// Whether any checks have failed.
  bool anyFailed;
};
} // namespace

void CriticalPathAnalysisPass::runOnOperation() {
  FModuleOp module = getOperation();
  anyFailed = false;

  LLVM_DEBUG(llvm::dbgs()
             << "executing CriticalPathAnalysisPass on " << module.getName().str() << ".\n");

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Port " << port.name << " of type " << port.type << "\n");
  }

  std::list<Operation*> worklist;
  double maxOpLatency = 0.0;
  Operation* criticalPathEnd = nullptr;

  struct ExprInitialLatencyEvaluator : public FIRRTLVisitor<ExprInitialLatencyEvaluator, double> {
    using FIRRTLVisitor<ExprInitialLatencyEvaluator, double>::visitExpr;

    double visitInvalidExpr(Operation *op) { return -1.0; }
    double visitUnhandledExpr(Operation *op) { return -1.0; }
    // constant operation have zero-latency
    double visitExpr(ConstantOp op) {return 0.0; }
  };

  auto initialEvaluator = ExprInitialLatencyEvaluator();


  // Check the results of each operation.
  module->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Found op  " << op->getName().getStringRef().str() << "\n");
    double latency = initialEvaluator.dispatchExprVisitor(op);
    LLVM_DEBUG(llvm::dbgs()
               << "latency for op  " << op->getName().getStringRef().str() << " has been evaluated to " << latency << "\n");
    // opsLatency[op] = latency;
    valuesLatency[op->getOpResult(0)] = latency;
    // enqueue operations for which latency was not determined
    if (latency < 0)
      worklist.push_back(op);
  });
  struct ExprLatencyEvaluator : public FIRRTLVisitor<ExprLatencyEvaluator, bool, llvm::DenseMap<Value, double>> {
    using FIRRTLVisitor<ExprLatencyEvaluator, bool, llvm::DenseMap<Value, double>>::visitExpr;
    using FIRRTLVisitor<ExprLatencyEvaluator, bool, llvm::DenseMap<Value, double>>::visitStmt;

    bool visitInvalidExpr(Operation *op, llvm::DenseMap<Value, double> latencyMap) { return false; }
    bool visitUnhandledExpr(Operation *op, llvm::DenseMap<Value, double> latencyMap) { return false; }
    // constant operation have zero-latency
    bool visitExpr(ConstantOp op, llvm::DenseMap<Value, double> latencyMap) {
      latencyMap[op->getOpResult(0)] = 0.0;
      return true;
     }

    bool visitExpr(SubfieldOp op, llvm::DenseMap<Value, double> latencyMap) {
      auto input = op.input();
      auto it = latencyMap.find(input);
      if (it == latencyMap.end()) return false;
      double latency = it->second;
      latencyMap[op->getOpResult(0)] = latency;
      return true;
    }
    bool visitExpr(XorPrimOp op, llvm::DenseMap<Value, double> latencyMap) {
      double maxLatency = 0.0;
      for (auto operand: op->getOperands()) {
        auto it = latencyMap.find(operand);
        if (it == latencyMap.end()) return false;
        double operandLatency = it->second;
        if (operandLatency > maxLatency)
          maxLatency = operandLatency;
      }
      const double latency_xor = 2.0;
      op->getOpResult(0).dump();
      latencyMap[op->getOpResult(0)] = maxLatency + latency_xor;
      return true;
    }
    bool visitExpr(AndPrimOp op, llvm::DenseMap<Value, double> latencyMap) {
      double maxLatency = 0.0;
      for (auto operand: op->getOperands()) {
        auto it = latencyMap.find(operand);
        if (it == latencyMap.end()) return false;
        double operandLatency = it->second;
        if (operandLatency > maxLatency)
          maxLatency = operandLatency;
      }
      const double latency_and = 2.0;
      op->getOpResult(0).dump();
      latencyMap[op->getOpResult(0)] = maxLatency + latency_and;
      return true;
    }
    bool visitStmt(ConnectOp op, llvm::DenseMap<Value, double> latencyMap) {
      auto it = latencyMap.find(op.src());
      if (it == latencyMap.end()) return false;
      double latency = it->second;
      latencyMap[op.dest()] = latency;
      return true;
    }
  };

  auto latencyEvaluator = ExprLatencyEvaluator();
  while (!worklist.empty()) {
    Operation* op = worklist.front();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing " << op->getName().getStringRef().str() << "\n");
    worklist.pop_front();
    bool latencyFound = latencyEvaluator.dispatchExprVisitor(op, valuesLatency);

    if (latencyFound) {
      auto it = valuesLatency.find(op->getOpResult(0));
      LLVM_DEBUG(llvm::dbgs()
                << "Found latency " << it->second << " for op " << op->getName().getStringRef().str() << "\n");

    } else {
      worklist.push_back(op);
    }

  }

  // LLVM_DEBUG(llvm::dbgs()
  //           << "critical path's end is " << criticalPathEnd->getName().getStringRef().str() << "\n");

  // looking for critical path

  markAllAnalysesPreserved();
}


std::unique_ptr<mlir::Pass> circt::firrtl::createCriticalPathAnalysisPass() {
  return std::make_unique<CriticalPathAnalysisPass>();
}

