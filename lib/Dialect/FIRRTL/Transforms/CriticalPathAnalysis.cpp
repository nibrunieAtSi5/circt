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
    Operation *nodeOp;
    // index of Value/OpResult in nodeOp corresponding to local value
    int resultIndex;
    /// pointer to previous node (upstream) in the critical path
    TimingPathNode *previousNode;
    /// pointer to the next node (downstream) in the critical path
    TimingPathNode *nextNode;
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

// return true if <val> is one of its block's argument
bool isBlockArgument(Value val) {
   return val.isa<BlockArgument>();
 }

class ModuleInfo {
public:
  FModuleOp module;
  bool isInputPort(Value val) {
    if (!isBlockArgument(val)) return false;
    auto argIndex = val.cast<BlockArgument>().getArgNumber();
    return !getModulePortInfo(module)[argIndex].isOutput();
  }
  bool isOutputPort(Value val) {
    if (!isBlockArgument(val)) return false;
    auto argIndex = val.cast<BlockArgument>().getArgNumber();
    return getModulePortInfo(module)[argIndex].isOutput();
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

int getMinWidthBinaryOp(Operation *op) {
  return std::min(getWidth(op->getOperand(0).getType()), getWidth(op->getOperand(1).getType()));
}

int getResultWidth(Operation *op) {
  return getWidth(op->getOpResult(0).getType());
}

bool isSignExtended(Operation* op) {
  auto type = op->getOpResult(0).getType();
  // todo: could check input type
  return type.cast<IntType>().isSigned();

}

// dummy timing model
// todo: translate logical-effort into proper evaluation rules
class TimingModel {
  public:
    static double getOpLatency(Operation* op) {
      return TypeSwitch<Operation*, double>(op)
            .template Case<XorPrimOp>([&](auto op) -> double { return 1.2;})
            .template Case<OrPrimOp, AndPrimOp>([&](auto op) -> double{ return 1.0;})
            .template Case<NotPrimOp>([&](auto op) -> double{ return 1.0;})

            .template Case<XorRPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getWidth(op.input().getType()))) * 1.2;})
            .template Case<AndRPrimOp, OrRPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getWidth(op.input().getType()))) * 1.0;})
            // should distinguish between operation with one operand constant and operand between two dynamic operands
            .template Case<EQPrimOp, NEQPrimOp>([&](auto op) -> double { return 1.2 + std::ceil(std::log2(getMinWidthBinaryOp(op))) * 1.0;})
            .template Case<AddPrimOp, SubPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getResultWidth(op))) * 2.2;})
            .template Case<GTPrimOp, LTPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getResultWidth(op))) * 2.2;})
            .template Case<GEQPrimOp, LEQPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getResultWidth(op))) * 2.2 + 1.0;})
            .template Case<ShlPrimOp, ShrPrimOp>([&](auto op) -> double { return 0.0;})
            .template Case<AsSIntPrimOp, AsUIntPrimOp, AsPassivePrimOp, AsNonPassivePrimOp, CvtPrimOp>([&](auto op) -> double { return 0.0;})
            .template Case<CatPrimOp>([&](auto op) -> double { return 0.0;})
            .template Case<MuxPrimOp>([&](auto op) -> double { return 2.0;})
            .template Case<PadPrimOp>([&](auto op) -> double { return isSignExtended(op) ? std::ceil(std::log2(op.amount())) * 0.5 : 0.0;}) // log2 for sign fanout
            .template Case<DShlPrimOp, DShrPrimOp>([&](auto op) -> double { return std::ceil(std::log2(getWidth(op.rhs().getType()))) * 2.0;})
            .Default([&](auto op) -> double { return -1.0; });
      }

};

bool isFlippedType(FIRRTLType type) {
  if (auto flip = type.dyn_cast<FlipType>())
    return true;
  return false;
}

bool isInputType(FIRRTLType type) {
  return !isFlippedType(type);
}

bool isInputType(Type type) {
  return isInputType(type.dyn_cast<FIRRTLType>());
}

bool isOutputType(FIRRTLType type) {
  return isFlippedType(type);
}

bool isOutputType(Type type) {
  return isOutputType(type.dyn_cast<FIRRTLType>());
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



/// Test recursively if the given <val>
// is a wire or a sub-wire (SubfieldOp ...)
 bool isWire(Value val) {
  if (!val) return false;
  Operation* op = val.getDefiningOp();
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
    .Case<SubfieldOp>([](auto op) { return isWire(op.input());})
    .Case<WireOp>([](auto) { return true;})
    .Default([](auto) { return false;});
 }

/// Extract the main WireOp Value from a wire or a sub-wire (SubfieldOp ...)
 Value getWireValue(Value val) {
  Operation* op = val.getDefiningOp();
  return TypeSwitch<Operation *, Value>(op)
    .Case<SubfieldOp>([](auto op) { return getWireValue(op.input());})
    .Case<WireOp>([&](auto) { return val;})
    .Default([](auto) { return Value();});
 }

 bool isRegister(Value val) {
  if (!val) return false;
  Operation* op = val.getDefiningOp();
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
    .Case<SubfieldOp>([](auto op) { return isRegister(op.input());})
    .Case<RegOp, RegResetOp>([](auto) { return true;})
    .Default([](auto) { return false;});
 }

 Value getRegisterValue(Value val) {
  Operation* op = val.getDefiningOp();
  return TypeSwitch<Operation *, Value>(op)
    .Case<SubfieldOp>([](auto op) { return getRegisterValue(op.input());})
    .Case<RegOp, RegResetOp>([&](auto) { return val;})
    .Default([](auto) { return Value();});
 }

bool isInputValue(ModuleInfo* moduleInfo, Value val) {
  if (!val) return false;
  if (isBlockArgument(val) && moduleInfo->isInputPort(val)) return true;
  Operation *op = val.getDefiningOp();
  if (!op) return false;
  return TypeSwitch<Operation*, bool>(op)
        .template Case<SubfieldOp>([&](auto op) -> bool {
             return isInputValue(moduleInfo, op.input());
            //return isBlockArgument(op.input()) && isInputField(moduleInfo, op.input().getType(), op.fieldname());
         })
        .Default([&](auto op) -> double { return false; });
}

bool isOutputValue(ModuleInfo *moduleInfo, Value dest) {
  if (!dest) return false;
  llvm::outs() << "testing is block argument ";
  dest.print(llvm::outs());
  llvm::outs() << "\n";
  if (isBlockArgument(dest) && moduleInfo->isOutputPort(dest)) {
    llvm::outs() << "   is output\n";
    return true;
  }
  llvm::outs() << "   is not output\n";
  Operation *op = dest.getDefiningOp();
  if (!op) return false;
  return TypeSwitch<Operation*, bool>(op)
        .template Case<SubfieldOp>([&](auto op) -> bool {
            // return isBlockArgument(op.input()) && !isInputField(op.input().getType(), op.fieldname());
            return isOutputValue(moduleInfo, op.input());
         })
        .Default([&](auto op) -> double { return false; });
}


void postProcessPaths(llvm::SmallVector<TimingPathNode*>& pathVectors) {
  for (auto pathEnd : pathVectors) {
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
      if (node && node->nodeOp) node->nodeOp->getOpResult(node->resultIndex).print(llvm::outs());
      llvm::outs() << "\n";

      index++;
    }
  }
}

struct ExprLatencyEvaluator : public FIRRTLVisitor<ExprLatencyEvaluator, bool> {
  public:
    // llvm::DenseMap<Operation, double> opsLatency;
    llvm::DenseMap<Value, TimingPathNode*> valuesLatency;
    llvm::SmallVector<TimingPathNode*> outputPaths;
    llvm::SmallVector<TimingPathNode*> fromRegPaths;
    llvm::SmallVector<TimingPathNode*> toRegPaths;
    ModuleInfo* moduleInfo;

    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitExpr;
    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitStmt;
    using FIRRTLVisitor<ExprLatencyEvaluator, bool>::visitDecl;

    // this callback ends the chain of visitInvalidExpr -> visitInvalidStmt -> visitInvalidDecl
    bool visitInvalidDecl(Operation *op) {
      llvm::errs() << "Unsupported operation: " << op->getName().getStringRef().str() << " \n";
      return false;
    }
    bool visitUnhandledExpr(Operation *op) {
      llvm::errs() << "Unsupported expression: " << op->getName().getStringRef().str() << " \n";
      return false;
    }
    bool visitUnhandledStmt(Operation *op) {
      llvm::errs() << "Unsupported statement: " << op->getName().getStringRef().str() << " \n";
      return false;
    }
    bool visitUnhandledDecl(Operation *op) {
      llvm::errs() << "Unsupported declaration: " << op->getName().getStringRef().str() << " \n";
      return false;
    }

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
      if (isInputValue(moduleInfo, input)) {

      } else {
        TimingPathNode* prePath = getStoredLatency(moduleInfo, input);
        if (prePath) return false;
        previous = prePath;
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
        TimingPathNode* prePath = getStoredLatency(moduleInfo, operand);
        if (!prePath) return false;
        double operandLatency = prePath->pathLatency;
        if (nullptr == longestPath || operandLatency > longestPath->pathLatency)
          longestPath = prePath;
      }
      valuesLatency[op->getOpResult(0)] = new TimingPathNode(opClassLatency, op, longestPath);
      return true;

    }


    bool visitExpr(XorPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AndPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(OrPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(AndRPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(OrRPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(XorRPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(NotPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(AddPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(SubPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(ShrPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(ShlPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(DShrPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(DShlPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(EQPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(NEQPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(LTPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(LEQPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(GTPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(GEQPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(CvtPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AsSIntPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AsUIntPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AsPassivePrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(AsNonPassivePrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(PadPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(CatPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }
    bool visitExpr(MuxPrimOp op) { return visitMultiAryBitwiseOp(op, TimingModel::getOpLatency(op)); }

    bool visitExpr(BitsPrimOp op) {
      TimingPathNode* prePath = getStoredLatency(moduleInfo, op.input());
      // todo: for now we discard the effect of the selection range
      if (!prePath) return false;
      TimingPathNode* localPath = new TimingPathNode(0.0, op, prePath);
      valuesLatency[op->getOpResult(0)] = localPath;
      return true;
    }

    // todo: should use Option-like mechanism to return both path and success flag rather (than inout arg)
    TimingPathNode* getStoredLatency(ModuleInfo *moduleInfo, Value val) {
      assert(moduleInfo);
      auto it = valuesLatency.find(val);
      if (it == valuesLatency.end() || it->second->pathLatency < 0) {
        if (isInputValue(moduleInfo, val)) {
            TimingPathNode* inputPath = new TimingPathNode(0.0, nullptr, nullptr);
            valuesLatency[val] = inputPath;
            return inputPath;
        }
        llvm::outs() << "could not find latency for Value " << val << " defined here: " << val.getDefiningOp() << "\n";
        return nullptr;
      }
      return it->second;
    }

    bool visitStmt(ConnectOp op) {
      Value srcOp;
      (void) isOutputValue(moduleInfo, op.dest());
      if (isWire(op.src())) srcOp = getWireValue(op.src());
      else srcOp = op.src();
      TimingPathNode* path = getStoredLatency(moduleInfo, srcOp);
      if (!path) return false;
      Value destOp;
      if (isWire(op.dest())) {
        // we squeeze wire as a single Value
        // todo: optimize by field
        destOp = getWireValue(op.dest());
      } else {
        destOp = op.dest();
      }
      TimingPathNode* localPath = new TimingPathNode(0.0, destOp.getDefiningOp(), path);
      valuesLatency[destOp] = localPath;
      if (isOutputValue(moduleInfo, destOp)) {
        llvm::outs() << "found connection to output ";
        destOp.print(llvm::outs());
        llvm::outs() << ".\n";
        outputPaths.push_back(localPath);
      } else if (isRegister(destOp)) {
        llvm::outs() << "found connection to register ";
        destOp.print(llvm::outs());
        llvm::outs() << ".\n";
        toRegPaths.push_back(localPath);
      }
      return true;
    }

    bool visitDecl(WireOp op){
      return true;
    }

    bool visitDecl(RegOp op){
      llvm::outs() << "declaring register ";
      op.result().print(llvm::outs());
      llvm::outs() << ".\n";
      valuesLatency[op.result()] = new TimingPathNode(0.0, op, nullptr);
      return true;
    }
};

class ModulePathTimingInfo {
public:
  StringRef startLabel;
  StringRef endLabel;
  double latency;

  ModulePathTimingInfo(StringRef startLabel,
                       StringRef endLabel,
                       double latency):
    startLabel(startLabel), endLabel(endLabel), latency(latency) {}
};

class ModuleTimingInfo {
  public:
    FModuleOp module;
    llvm::DenseMap<StringRef, ModulePathTimingInfo> pathFromStart;
    llvm::DenseMap<StringRef, ModulePathTimingInfo> pathFromEnd;
};


/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CriticalPathAnalysisPass : public CriticalPathAnalysisBase<CriticalPathAnalysisPass> {
  llvm::DenseMap<StringRef, ModuleTimingInfo> moduleTimings;

  void runOnOperation() override;

};
} // namespace



void CriticalPathAnalysisPass::runOnOperation() {
  FModuleOp module = getOperation();

  ModuleTimingInfo moduleTiming;
  ModuleInfo moduleInfo;
  moduleInfo.module = module;

  llvm::outs() << "executing CriticalPathAnalysisPass on " << module.getName().str() << ".\n";

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    //LLVM_DEBUG(llvm::dbgs()

    llvm::outs() << "Port " << port.getName() << " of type " << port.type << " isOutput " << port.isOutput() << "\n";
  }

  auto latencyEvaluator = ExprLatencyEvaluator();
  latencyEvaluator.moduleInfo = &moduleInfo;

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

  while (false && !worklist.empty()) {
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

  postProcessPaths(latencyEvaluator.outputPaths);
  // postProcessPaths(latencyEvaluator.toRegPaths);


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

