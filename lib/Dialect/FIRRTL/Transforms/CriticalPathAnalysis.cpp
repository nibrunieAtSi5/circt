//===- CriticalPathAnalysis.cpp - Analyse FIRRTL module critical paths ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CriticalPathAnalysis pass.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/Debug.h"

#include <iomanip>
#include <sstream>

#include <list>
#include <cmath>

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
    // index of Value/OpResult in nodeOp corresponding to local value
    int resultIndex;
    /// pointer to previous node (upstream) in the critical path
    TimingPathNode* previousNode;
    /// pointer to the next node (downstream) in the critical path
    TimingPathNode* nextNode;
    /// latency of the critical path ending at this node (including
    //  this node local latency)
    double pathLatency;

    TimingPathNode(): nodeLatency(0.0), nodeOp(nullptr), resultIndex(-1), previousNode(nullptr),
                      nextNode(nullptr), pathLatency(0.0) {}

    TimingPathNode(double latency, Operation* op, TimingPathNode* previous, int index=0):
      nodeLatency(latency), resultIndex(index), previousNode(previous), nextNode(nullptr) {
        nodeOp = op;
        pathLatency = (previous ? previousNode->pathLatency : 0.0) + nodeLatency;
        if (previous) previous->nextNode = this;
      }
};


/// Convert double precision value to string (used when injecting latency
//  measurement into llvm::outs() )
std::string doubleToString(double value, int prec=3) {
  std::ostringstream streamObj;
  streamObj << std::fixed << std::setprecision(prec) << value;
  return streamObj.str();
}

/// Extract a FIRRTL Type's width (if any else return -1)
int getWidth(Type type) {
  return TypeSwitch<Type, int>(type)
    .template Case<IntType>([&](auto type) {
        return type.getWidth();
     })
    .Default([&](auto type) -> int { return -1;});
}

class TimingModel {
  public:
    static double getOpLatency(Operation* op) {
      return TypeSwitch<Operation*, double>(op)
            .template Case<XorPrimOp>([&](auto op) -> double { return 1.2;})
            .template Case<OrPrimOp, AndPrimOp>([&](auto op) -> double{ return 1.0;})
            .template Case<NotPrimOp>([&](auto op) -> double{ return 1.0;})

            .template Case<XorRPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getWidth(op.input().getType()))) * 1.2;})
            .template Case<AndRPrimOp, OrRPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getWidth(op.input().getType()))) * 1.0;})
            .Default([&](auto op) -> double { return -1.0; });
      }

};

bool isFlippedType(FIRRTLType type) {
  if (auto flip = type.dyn_cast<FlipType>())
    return true;
  return false;
}

// check if field <name> in aggregate format <type> is an input or not
bool isInputField(FIRRTLType type, StringRef name) {
   LLVM_DEBUG(llvm::dbgs() << "isInputField " << type << " " << name << "\n");
  if (auto flip = type.dyn_cast<FlipType>()) {
     LLVM_DEBUG(llvm::dbgs() << "Input field is flipped\n");
    return !isInputField(flip.getElementType(), name);
  }
  return TypeSwitch<FIRRTLType, bool>(type)
        .Case<BundleType>([&](auto bundle) {
          // Otherwise, we have a bundle type.  Break it down.
          for (auto &elt : bundle.getElements()) {
            if (elt.name.getValue() == name) {
              return !isFlippedType(elt.type);
            }
          }
          LLVM_DEBUG(llvm::dbgs() << "Input field not found\n");
          // todo: field not found
          return false;
        })
        .Default([&](auto) {
          LLVM_DEBUG(llvm::dbgs() << "Input field default case\n");
          return false;
        });

    LLVM_DEBUG(llvm::dbgs() << "Input field not matched by TypeSwitch\n");
    return false;
}

bool isInputField(Type type, StringRef name) {
    return isInputField(type.dyn_cast<FIRRTLType>(), name);
}


 bool isBlockArgument(Value val) {
   return val.isa<BlockArgument>();
 }

bool isOutputValue(Value dest) {
  return TypeSwitch<Operation*, bool>(dest.getDefiningOp())
        .template Case<SubfieldOp>([&](auto op) -> bool {
            return isBlockArgument(op.input()) && !isInputField(op.input().getType(), op.fieldname());
         })
        .Default([&](auto op) -> double { return false; });
}
struct ExprLatencyEvaluator : public FIRRTLVisitor<ExprLatencyEvaluator, bool> {
  public:
    // llvm::DenseMap<Operation, double> opsLatency;
    llvm::DenseMap<Value, TimingPathNode*> valuesLatency;
    llvm::SmallVector<TimingPathNode*> outputPaths;

    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitExpr;
    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitStmt;

    // this callback ends the chain of visitInvalidExpr -> visitInvalidStmt -> visitInvalidDecl
    bool visitInvalidDecl(Operation *op) {
      llvm::errs() << "Unsupported operation: " << op->getName().getStringRef().str() << " \n";
      return false;
    }
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
        LLVM_DEBUG(llvm::dbgs() << "field type is flipped=" << isInput << "\n.");
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
      valuesLatency[op->getOpResult(0)] = new TimingPathNode(opClassLatency, op, longestPath);
      return true;

    }


    bool visitExpr(XorPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AndPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(OrPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitStmt(ConnectOp op) {
      auto it = valuesLatency.find(op.src());
    LLVM_DEBUG(llvm::dbgs()
               << "Processing connect\n");
      if (it == valuesLatency.end() || it->second->pathLatency < 0) return false;
      TimingPathNode* localPath = new TimingPathNode(0.0, op.dest().getDefiningOp(), (it->second));
      valuesLatency[op.dest()] = localPath;
      if (isOutputValue(op.dest())) {
        outputPaths.push_back(localPath);
      }
      return true;
    }
};

/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CriticalPathAnalysisPass : public CriticalPathAnalysisBase<CriticalPathAnalysisPass> {
  void runOnOperation() override;

};
} // namespace

void CriticalPathAnalysisPass::runOnOperation() {
  FModuleOp module = getOperation();

  llvm::outs() << "executing CriticalPathAnalysisPass on " << module.getName().str() << ".\n";

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Port " << port.name << " of type " << port.type << "\n");
  }

  auto latencyEvaluator = ExprLatencyEvaluator();

  // list of Operation node awaiting processing
  std::list<Operation*> worklist;

  // temporary determined critical path end node
  TimingPathNode* criticalPathEnd = nullptr;

  // Check the results of each operation.
  module->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Found op  " << op->getName().getStringRef().str() << "\n");
    if (auto mod = dyn_cast<FModuleOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                << "Found to discard: " << op->getName().getStringRef().str() << "\n");
      return;
    }
    bool latencyFound = latencyEvaluator.dispatchExprVisitor(op);
    LLVM_DEBUG(llvm::dbgs()
               << "latency for op  " << op->getName().getStringRef().str() << " has been evaluated.\n");
    // enqueue operations for which latency was not determined
    if (!latencyFound)
      worklist.push_back(op);
    else {
      auto it = latencyEvaluator.valuesLatency.find(op->getOpResult(0));
      if (it != latencyEvaluator.valuesLatency.end()) {
        if (nullptr == criticalPathEnd || criticalPathEnd->pathLatency < it->second->pathLatency)
        criticalPathEnd = (it->second);
        LLVM_DEBUG(llvm::dbgs()
                  << "Found latency " << it->second->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      }
    }
  });

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
        LLVM_DEBUG(llvm::dbgs()
                  << "Found latency " << it->second->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      }
    } else {
      worklist.push_back(op);
    }
  }

  // todo: need a more canonical part to display result info
  // critical path traversal
  // start by looking for path start
  llvm::outs() << "Rewiding critical paths:\n";
  for (auto pathEnd : latencyEvaluator.outputPaths) {
    std::list<TimingPathNode*> localPath;
    TimingPathNode* criticalPathStart = pathEnd;
    while (criticalPathStart != nullptr && criticalPathStart->previousNode != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "  Found new node." << criticalPathStart << " " << criticalPathStart->nodeOp << "\n");
     localPath.push_front(criticalPathStart);
      criticalPathStart = criticalPathStart->previousNode;
    }
    if (criticalPathStart) localPath.push_front(criticalPathStart);
    int index = 0;
    llvm::outs() << "critical path is:\n";
    for (auto node : localPath) {
      // result display
      llvm::outs() << "#" << index << ": " << doubleToString(node->nodeLatency) << " " << doubleToString(node->pathLatency) << " ";
      node->nodeOp->getOpResult(node->resultIndex).print(llvm::outs());
      llvm::outs() << "\n";

      index++;
    }
  }


  // cleaning memory
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

