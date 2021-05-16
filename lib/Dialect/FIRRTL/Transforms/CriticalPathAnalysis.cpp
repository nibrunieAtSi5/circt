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
#include <iostream>

#include <list>

#define DEBUG_TYPE "firrtl-critical-path"

using namespace circt;
using namespace firrtl;

namespace {


// node in a TimingPath
class TimingPathNode {
  public:
    /// local node latency
    double nodeLatency;
    Operation* nodeOp;
    /// pointer to previous node (upstream) in the critical path
    TimingPathNode* previousNode;
    /// pointer to the next node (downstream) in the critical path
    TimingPathNode* nextNode;
    /// latency of the critical path ending at this node (including
    //  this node local latency)
    double pathLatency;

    TimingPathNode(): nodeLatency(0.0), nodeOp(nullptr), previousNode(nullptr),
                      nextNode(nullptr), pathLatency(0.0) {}

    TimingPathNode(double latency, Operation* op, TimingPathNode* previous):
      nodeLatency(latency), previousNode(previous), nextNode(nullptr) {
        llvm::outs() << "registering node " << op << "\n";
        nodeOp = op;
        pathLatency = (previous ? previousNode->pathLatency : 0.0) + nodeLatency;
        if (previous) previous->nextNode = this;
      }
};

bool isFlippedType(FIRRTLType type) {
  if (auto flip = type.dyn_cast<FlipType>())
    return true;
  return false;
}

// check if field <name> in aggregate format <type> is an input or not
bool isInputField(FIRRTLType type, StringRef name) {
  if (auto flip = type.dyn_cast<FlipType>()) {
    return !isInputField(flip.getElementType(), name);
  }
  TypeSwitch<FIRRTLType>(type)
        .Case<BundleType>([&](auto bundle) {
          // Otherwise, we have a bundle type.  Break it down.
          for (auto &elt : bundle.getElements()) {
            if (elt.name.getValue() == name) {
              return isFlippedType(elt.type);
            }
          }
          // todo: field not found
          return false;
        })
        .Default([&](auto) {
          return false;
        });

    return false;
}


/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CriticalPathAnalysisPass : public CriticalPathAnalysisBase<CriticalPathAnalysisPass> {
  void runOnOperation() override;


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
    // valuesLatency[op->getOpResult(0)] = latency;
    // enqueue operations for which latency was not determined
    if (latency < 0)
      worklist.push_back(op);
  });
  struct ExprLatencyEvaluator : public FIRRTLVisitor<ExprLatencyEvaluator, bool> {
  public:
    // llvm::DenseMap<Operation, double> opsLatency;
    llvm::DenseMap<Value, TimingPathNode*> valuesLatency;

    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitExpr;
    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitStmt;

    // this callback ends the chain of visitInvalidExpr -> visitInvalidStmt -> visitInvalidDecl
    bool visitInvalidDecl(Operation *op) { return false; }
    bool visitUnhandledExpr(Operation *op) { return false; }
    // constant operation have zero-latency
    bool visitExpr(ConstantOp op) {
      valuesLatency[op->getOpResult(0)] = new TimingPathNode(0.0, op, nullptr);
      return true;
     }

    bool visitExpr(SubfieldOp op) {
      auto input = op.input();
      auto field = op.fieldname();
      LLVM_DEBUG(llvm::dbgs()
                << "SubField's fieldname is " << field << " \n");
      TimingPathNode* previous = nullptr;
      if (input.isa<BlockArgument>()) {

        LLVM_DEBUG(llvm::dbgs()
                  << "SubField's input is a BlockArgument. " << input.getType() << " \n");
        bool isInput = isInputField(input.getType().dyn_cast<FIRRTLType>(), field);
        //if (auto flipType = input.getType().dyn_cast<firrtl::FlipType>()) {
          LLVM_DEBUG(llvm::dbgs() << "field type is flipped=" << isInput << "\n.");
        //}
      } else {
        auto it = valuesLatency.find(input);
        if (it == valuesLatency.end() || it->second->pathLatency < 0) return false;
        previous = (it->second);
      }
      TimingPathNode* pathNode = new TimingPathNode(0.0, op, previous);
      LLVM_DEBUG(llvm::dbgs()
                << "Inner Found latency " << pathNode->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      valuesLatency[op->getOpResult(0)] = pathNode;
      return true;
    }

    // visit generic multi-ary bitwise operation (e.g. Xor, And ...)
    bool visitMultiAryBitwiseOp(Operation* op, const double opClassLatency) {
      TimingPathNode* longestPath = nullptr;
      for (auto operand: op->getOperands()) {
        auto it = valuesLatency.find(operand);
        if (it == valuesLatency.end() || it->second->pathLatency < 0) return false;
        double operandLatency = it->second->pathLatency;
        if (nullptr == longestPath || operandLatency > longestPath->pathLatency)
          longestPath = (it->second);
      }
      op->getOpResult(0).dump();
      valuesLatency[op->getOpResult(0)] = new TimingPathNode(opClassLatency, op, longestPath);
      return true;

    }

    bool visitExpr(XorPrimOp op) {
      const double latency_xor = 2.0;
      return visitMultiAryBitwiseOp(op, latency_xor);
    }
    bool visitExpr(AndPrimOp op) {
      const double latency_and = 2.0;
      return visitMultiAryBitwiseOp(op, latency_and);
    }
    bool visitStmt(ConnectOp op) {
      auto it = valuesLatency.find(op.src());
    LLVM_DEBUG(llvm::dbgs()
               << "Processing connect\n");
      op.src().dump();
      if (it == valuesLatency.end() || it->second->pathLatency < 0) return false;
      valuesLatency[op.dest()] = new TimingPathNode(0.0, op.dest().getDefiningOp(), (it->second));
      return true;
    }
  };

  auto latencyEvaluator = ExprLatencyEvaluator();
  TimingPathNode* criticalPathEnd = nullptr;
  // int watchdog = 20;
  while (!worklist.empty()) {
    Operation* op = worklist.front();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing " << op->getName().getStringRef().str() << "\n");
    worklist.pop_front();

    if (auto mod = dyn_cast<FModuleOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                << "Found to discard: " << op->getName().getStringRef().str() << "\n");
      continue;
    }

    bool latencyFound = latencyEvaluator.dispatchExprVisitor(op);

    if (latencyFound) {
      auto it = latencyEvaluator.valuesLatency.find(op->getOpResult(0));
      if (it == latencyEvaluator.valuesLatency.end()) {
        LLVM_DEBUG(llvm::dbgs()
                  << "invalid latency for op " << op->getName().getStringRef().str() << "\n");

      } else {
        if (nullptr == criticalPathEnd || criticalPathEnd->pathLatency < it->second->pathLatency)
        criticalPathEnd = (it->second);
        LLVM_DEBUG(llvm::dbgs()
                  << "Found latency " << it->second->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      }

    } else {
      worklist.push_back(op);
    }

    // if (watchdog-- < 0) break;

  }
  for (auto it : latencyEvaluator.valuesLatency) {
    LLVM_DEBUG(llvm::dbgs()
               << "latency is " << it.second->pathLatency << "\n");

  }

  // todo: need a more canonical part to display result info
  // critical path traversal
  // start by looking for path start
  TimingPathNode* criticalPathStart = criticalPathEnd;
  llvm::outs() << "Rewiding critical path:\n";
  while (criticalPathStart != nullptr && criticalPathStart->previousNode != nullptr) {
    llvm::outs() << "  Found new node." << criticalPathStart << " " << criticalPathStart->nodeOp << "\n";
    criticalPathStart = criticalPathStart->previousNode;
  }
  llvm::outs() << "critical path is:\n";
  //criticalPathEnd->nodeOp->getOpResult(0).dump();
  //criticalPathStart->nodeOp->getOpResult(0).dump();
  for (int index = 0; ; index++) {
    if (criticalPathStart == nullptr) break;
    llvm::outs() << "#" << index << ": ";
    criticalPathStart->nodeOp->getOpResult(0).print(llvm::outs());
    llvm::outs() << "\n";
    // << criticalPathStart->nodeOp->getName().getStringRef().str() << "\n";
    criticalPathStart = criticalPathStart->nextNode;
  }

  LLVM_DEBUG(llvm::dbgs() << "cleaning memory.\n");

  for (auto it : latencyEvaluator.valuesLatency) {
    delete it.second;
  }

  // todo: keep ?
  markAllAnalysesPreserved();
}


std::unique_ptr<mlir::Pass> circt::firrtl::createCriticalPathAnalysisPass() {
  return std::make_unique<CriticalPathAnalysisPass>();
}

