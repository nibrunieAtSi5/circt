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

/// Convert double precision value to string (used when injecting latency
//  measurement into llvm::outs() )
std::string doubleToString(double value, int prec=3) {
  std::ostringstream streamObj;
  streamObj << std::fixed << std::setprecision(prec) << value;
  return streamObj.str();
}

// return true if <val> is one of its block's argument
bool isBlockArgument(Value val) { return val.isa<BlockArgument>(); }

// test if <val> is a Register operation (directly or recursively)
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

// test if <val> is a constant operation (directly only)
bool isConstant(Value val) {
  if (!val) return false;
  Operation* op = val.getDefiningOp();
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
    .Case<ConstantOp>([](auto) { return true;})
    .Default([](auto) { return false;});
}

// predicate testing if an op can be discarded while building critical paths
bool discardableOp(Operation* op) {
  return TypeSwitch<Operation *, bool>(op)
  .Case<FModuleOp, PrintFOp, StopOp>([](auto op) { return true;})
  .Default([](auto) { return false;});
}

// check if <val> can be a timing path termination (I/O or register)
bool isPathTerminationValue(Value val) {
  return isBlockArgument(val) || isRegister(val);
}

class ModuleTimingInfo;
class TimingPath;
StringRef getModuleNameFromInfo(ModuleTimingInfo*);

// node in a TimingPath
//
// This object stores the information for a "step" in a timing path
// - link to previous and next nodes
// - local operation and associated latency information
template <class T>
class TimingPathNode {
  public:
    /// local node latency
    double nodeLatency;
    T nodeOp;
    // index of Value/OpResult in nodeOp corresponding to local value
    int resultIndex;
    /// pointer to previous node (upstream) in the critical path
    TimingPathNode<T> *previousNode;
    /// pointer to the next node (downstream) in the critical path
    TimingPathNode<T> *nextNode;
    /// latency of the critical path ending at this node (including
    //  this node local latency)
    double pathLatency;
    // pointer towards the module where this path is located
    ModuleTimingInfo* moduleTimingInfo;

    // the timing path up to and including this node is only involving
    // constant nodes and operations on recursive constants
    // and thus should not be prioritized when building a critical-path
    bool constant;

    TimingPathNode(): nodeLatency(0.0), nodeOp(nullptr), resultIndex(-1), previousNode(nullptr),
                      nextNode(nullptr), pathLatency(0.0), moduleTimingInfo(nullptr), constant(false) {}

    TimingPathNode(TimingPathNode& _rvalue) = default;

    TimingPathNode(double latency, T op, TimingPathNode<T>* previous, int index=0, ModuleTimingInfo* _module=nullptr, bool cst=false):
      nodeLatency(latency), resultIndex(index), previousNode(previous), nextNode(nullptr), moduleTimingInfo(_module), constant(cst) {
        nodeOp = op;
        pathLatency = (previous ? previousNode->pathLatency : 0.0) + nodeLatency;
        if (previous) previous->nextNode = this;
    }

    // determine if this node is a candidate to terminate (start/finish) a timing
    // path (e.g. being a register, being a top-module input/output ...)
    bool isPathTerminationNode();

    // determine the first node (start point) of a critical timing path going through
    // this node
    // @param discardCurrent does not consider current not as a potential candidate
    TimingPathNode* getPathFirstNode(bool discardCurrent=false) {
      TimingPathNode* current = this;
      while (current->previousNode && (!current->isPathTerminationNode() || (current == this && discardCurrent))) current = current->previousNode;
      return current;
    }

    // determine the last node (end point) of a critical timing path going through
    // this node
    // @param discardCurrent does not consider current not as a potential candidate
    TimingPathNode* getPathLastNode(bool discardCurrent=false) {
      TimingPathNode* current = this;
      while (current->nextNode && (!current->isPathTerminationNode() || (current == this && discardCurrent))) current = current->nextNode;
      return current;
    }

    // copy path starting from this node and rewinding upstream
    // in the copied path this node copy will have no nextNode
    TimingPathNode* copyPathUpstream(TimingPathNode* next=nullptr) {
      TimingPathNode* newNode = new TimingPathNode(*this);
      newNode->nextNode = next;
      if (previousNode) newNode->previousNode = previousNode->copyPathUpstream(newNode);
      return newNode;
    }

    // copy path starting from this node going downstream
    // in the copied path this node copy will have no previousNode
    TimingPathNode* copyPathDownstream() {
      TimingPathNode* newNode = new TimingPathNode(this);
      newNode->previousNode = nullptr;
      if (nextNode) newNode->nextNode = nextNode->copyPathUpstream();
      return newNode;
    }

    // display node information on \p stream
    void print(raw_ostream& stream, bool displayLoc=false);
};

// specialization for Value based timing-path node
using TimingPathNodeOp = TimingPathNode<Value>;


// extract the port name associated with \p val assuming
// it is a block argument
StringRef getModulePortName(Value val) {
  assert(isBlockArgument(val) && "val must be a BlockArgument in getModulePortName");
  FModuleOp module = cast<FModuleOp>(val.cast<BlockArgument>().getOwner()->getParentOp());
  auto argIndex = val.cast<BlockArgument>().getArgNumber();
  auto portName = getModulePortInfo(module)[argIndex].getName();
  return portName;
}

template<> void TimingPathNodeOp::print(raw_ostream& stream, bool displayLoc) {
  // todo: clean, mixing stream and diagnostic handler with emitRemark is a bad idea
  if (nodeOp) {
     if (isBlockArgument(nodeOp)) {
       stream << getModulePortName(nodeOp);
     } else nodeOp.getDefiningOp()->emitRemark("CP step"); // print(llvm::outs());
    if (displayLoc) stream << " " << nodeOp.getLoc();
    stream << "\n";
  }
}

template<> bool TimingPathNodeOp::isPathTerminationNode(){
  return isPathTerminationValue(nodeOp);
}


// Class to store a complete timing-path
class TimingPath {
public:
  // path starting node
  TimingPathNodeOp* startPoint;
  // path finish node
  TimingPathNodeOp* endPoint;
  double latency;
  // list of path nodes from start to finish
  std::list<TimingPathNodeOp*> nodes;
  TimingPath(TimingPathNodeOp* node, ModuleTimingInfo* moduleTimingInfo) {
    endPoint = node->getPathLastNode();
    TimingPathNodeOp* current = endPoint;
    while (current) {
      // setting module associated with path's node
      // (only when unset)
      if (!current->moduleTimingInfo) current->moduleTimingInfo = moduleTimingInfo;
      // todo: need to go from node to previous as going downstream seems broken
      // but push_front / vector is certainly costly
      nodes.push_front(current);
      current = current->previousNode;
    }
    startPoint = nodes.front();
    latency = endPoint->pathLatency;
  }

  void print(raw_ostream& stream, bool displayLoc=false) {
    int index = 0;
    stream << "critical path:\n";
    stream << "    Start point: ";
    startPoint->print(stream);
    stream << "    End   point: ";
    endPoint->print(stream);
    stream << "    Latency:     " << doubleToString(latency) << "\n";
    for (auto node : nodes) {
      // result display
      stream << "#" << index << ": " << doubleToString(node->nodeLatency) << " " << doubleToString(node->pathLatency) << " ";
      if (node && node->nodeOp) {
        stream << (node->moduleTimingInfo ? (getModuleNameFromInfo(node->moduleTimingInfo) + " ") : "");
         if (isBlockArgument(node->nodeOp)) {
           stream << getModulePortName(node->nodeOp);
         } else node->nodeOp.print(stream);
        if (displayLoc) stream << "\n  " << node->nodeOp.getLoc();
        stream << "\n";
      }
      index++;
    }
  }

  // check if this path start at a module input
  bool startAtInput() {
    return isBlockArgument(startPoint->nodeOp) && foldFlow(startPoint->nodeOp) == Flow::Source;
  }

  // check if this path start at a module register
  bool startAtRegister() {
    return isRegister(startPoint->nodeOp);
  }

  // check if this path start at a constant
  // todo: this should not happen, path starting at constant should be discarded
  bool startAtConstant() {
    return isConstant(startPoint->nodeOp);
  }

  // check if this module ends at a module output
  bool endAtOutput() {
    // todo: output may also be Flow::Duplex if they are re-used as expression source
    // in the same module
    return isBlockArgument(endPoint->nodeOp) && foldFlow(endPoint->nodeOp) == Flow::Sink;
  }

  // check if this module ends at a module register
  bool endAtRegister() {
    return isRegister(endPoint->nodeOp);
  }

  // return the starting label of the path (if any)
  StringRef startLabel() {
    if (startAtRegister()) return "<register>";
    if (startAtConstant()) return "<constant>";
    else return getModulePortName(startPoint->nodeOp);
  }

  // return the finish label of the path (if any)
  StringRef endLabel() {
    if (endAtRegister()) return "<register>";
    else return getModulePortName(endPoint->nodeOp);
  }
};

class ModulePathTimingInfo {
public:
  StringRef startLabel;
  StringRef endLabel;
  TimingPath path;

  ModulePathTimingInfo() = default;

  ModulePathTimingInfo(StringRef startLabel,
                       StringRef endLabel,
                       TimingPath _path):
    startLabel(startLabel), endLabel(endLabel), path(_path) {}
};

class ModuleTimingInfo {
  public:
    FModuleOp module;
    SmallVector<ModulePortInfo, 8> portInfo;
    ModuleTimingInfo(FModuleOp mod) : module(mod) {
      portInfo = getModulePortInfo(mod);
    }
    // llvm::DenseMap<StringRef, ModulePathTimingInfo> pathFromStart; -> no unicity of critical-path seen from start point
    llvm::DenseMap<StringRef, ModulePathTimingInfo*> pathFromOutput;
    llvm::SmallVector<TimingPath> pathToRegs;

    ModulePathTimingInfo* getPathFromOutput(StringRef portName) {
      for (auto label_path : pathFromOutput) {
        LLVM_DEBUG(llvm::dbgs() << "label_path.first=" << label_path.first.str() << " portName.str=" << portName.str() << "\n");
        if (label_path.first.str() == portName.str()) return label_path.second;
      }
      return nullptr;
    }

    // register the list of critical module paths which ends at a module output
    void registerOutputPath(llvm::SmallVector<TimingPathNodeOp*> &pathVector) {
      for (auto pathEnd: pathVector) {
        TimingPath localPath(pathEnd, this);
        LLVM_DEBUG(llvm::dbgs()
                    << "registering path for module " << module.getName() << " ending at " << localPath.endLabel() << " " << localPath.endAtOutput() << "\n");
        if (localPath.endAtOutput()) {
          pathFromOutput[localPath.endLabel()] = new ModulePathTimingInfo(localPath.startLabel(), localPath.endLabel(), localPath);
        }
      }
    }

    // register the list of critical module paths which ends at a module register
    void registerPathToRegs(llvm::SmallVector<TimingPathNodeOp*> &pathVector) {
      for (auto pathEnd: pathVector) {
        TimingPath localPath(pathEnd, this);
        pathToRegs.push_back(localPath);
      }
    }

    void displayPaths(bool displayLoc=false) {
        llvm::outs() << "Critical path for module" << module.getName() << "\n";
        llvm::outs() << "== path from output" << module.getName() << "\n";
        for (auto it : pathFromOutput) {
          it.second->path.print(llvm::outs(), displayLoc);
        }
        llvm::outs() << "== path from register" << module.getName() << "\n";
        for (auto path : pathToRegs) {
          path.print(llvm::outs(), displayLoc);
        }
    }
};

StringRef getModuleNameFromInfo(ModuleTimingInfo* moduleInfo) {
  return moduleInfo->module.getName();
}

// Map of timing info on multiple modules
class MapModuleTimingInfo {
public:
  DenseMap<StringRef, ModuleTimingInfo*> moduleMap;

  // todo: clean bad memory management: moduleTimingInfo is allocated somewhere and never free-ed
  void registerModuleTimingInfo(StringRef moduleName, ModuleTimingInfo* moduleTimingInfo) {
    moduleMap[moduleName] = moduleTimingInfo;
  }
  ModuleTimingInfo* getModuleTimingInfoByName(StringRef moduleName) {
    return moduleMap[moduleName];
  }
};


class ModuleInfo {
public:
  // local module
  FModuleOp module;
  // determine if <val> is an input port of <module>
  bool isInputPort(Value val) {
    if (!isBlockArgument(val)) return false;
    auto argIndex = val.cast<BlockArgument>().getArgNumber();
    return !getModulePortInfo(module)[argIndex].isOutput();
  }

  // determine if <val> is an output port of <module>
  bool isOutputPort(Value val) {
    if (!isBlockArgument(val)) return false;
    auto argIndex = val.cast<BlockArgument>().getArgNumber();
    return getModulePortInfo(module)[argIndex].isOutput();
  }

  // return the name of a port (if <val> is indeed a port)
  StringRef getPortName(Value val) {
    if (!isBlockArgument(val)) return "<no-input>";
    auto argIndex = val.cast<BlockArgument>().getArgNumber();
    return getModulePortInfo(module)[argIndex].getName();

  }
};


/// Extract a FIRRTL Type's width (if any else return -1)
int getWidth(Type type) {
  return TypeSwitch<Type, int>(type)
    .template Case<IntType>([&](auto type) {
        return type.getWidth();
     })
    .Default([&](auto type) -> int { return -1;});
}

// get the minimal width of the operands of a 2-input operations
int getMinWidthBinaryOp(Operation *op) {
  return std::min(getWidth(op->getOperand(0).getType()), getWidth(op->getOperand(1).getType()));
}

// get the width of the first result of <op>
int getResultWidth(Operation *op) {
  return getWidth(op->getOpResult(0).getType());
}

// is the type of the first result of <op> a signed integer type
// (assuming it is an integer type)
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


// extract the RegOp associated to val
 Value getRegisterValue(Value val) {
  Operation* op = val.getDefiningOp();
  return TypeSwitch<Operation *, Value>(op)
    .Case<SubfieldOp>([](auto op) { return getRegisterValue(op.input());})
    .Case<RegOp, RegResetOp>([&](auto) { return val;})
    .Default([](auto) { return Value();});
 }

// test if <val> is an input port
bool isInputValue(Value val) {
  if (!val) return false;
  auto kind = getDeclarationKind(val);
  if (kind != DeclKind::Port) return false;
  return foldFlow(val) == Flow::Source;
}

// test if <val> is an output port
bool isOutputValue(Value val) {
  if (!val) return false;
  auto kind = getDeclarationKind(val);
  if (kind != DeclKind::Port) return false;
  return foldFlow(val) == Flow::Sink;
}



// Specialization of FIRRTLVisitor to visit all the operation nodes
// and evaluate latency node by node
struct ExprLatencyEvaluator : public FIRRTLVisitor<ExprLatencyEvaluator, bool> {
  public:
    // llvm::DenseMap<Operation, double> opsLatency;
    llvm::DenseMap<Value, TimingPathNodeOp*> valuesLatency;
    llvm::SmallVector<TimingPathNodeOp*> outputPaths;
    llvm::SmallVector<TimingPathNodeOp*> fromRegPaths;
    llvm::SmallVector<TimingPathNodeOp*> toRegPaths;
    ModuleInfo* moduleInfo;
    MapModuleTimingInfo* moduleMap;

    ExprLatencyEvaluator(ModuleInfo* _moduleInfo, MapModuleTimingInfo* _map): moduleInfo(_moduleInfo), moduleMap(_map) {

    }

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
      valuesLatency[op->getOpResult(0)] = new TimingPathNodeOp(0.0, op->getOpResult(0), nullptr, true /* constant */);
      return true;
     }

    bool visitExpr(SubfieldOp op) {
      auto input = op.input();
      auto field = op.fieldname();
      LLVM_DEBUG(llvm::dbgs()
                << "SubField's fieldname is " << field << " \n");
      TimingPathNodeOp* previous = nullptr;
      if (isInputValue(input)) {

      } else {
        TimingPathNodeOp* prePath = getStoredLatency(input);
        if (prePath) return false;
        previous = prePath;
      }
      TimingPathNodeOp* pathNode = new TimingPathNodeOp(0.0, op->getOpResult(0), previous);
      LLVM_DEBUG(llvm::dbgs()
                << "Inner Found latency " << pathNode->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      valuesLatency[op->getOpResult(0)] = pathNode;
      return true;
    }

    // visit generic multi-ary bitwise operation (e.g. Xor, And ...)
    bool visitMultiAryBitwiseOp(Operation* op, const double opClassLatency) {
      TimingPathNodeOp* longestPath = nullptr;
      for (auto operand: op->getOperands()) {
        TimingPathNodeOp* prePath = getStoredLatency(operand);
        if (!prePath) return false;
        double operandLatency = prePath->pathLatency;
        if (nullptr == longestPath || operandLatency > longestPath->pathLatency || longestPath->constant)
          longestPath = prePath;
      }
      valuesLatency[op->getOpResult(0)] = new TimingPathNodeOp(opClassLatency, op->getOpResult(0), longestPath);
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
      TimingPathNodeOp* prePath = getStoredLatency(op.input());
      // todo: for now we discard the effect of the selection range
      if (!prePath) return false;
      TimingPathNodeOp* localPath = new TimingPathNodeOp(0.0, op->getOpResult(0), prePath);
      valuesLatency[op->getOpResult(0)] = localPath;
      return true;
    }

    // todo: should use Option-like mechanism to return both path and success flag rather (than inout arg)
    TimingPathNodeOp* getStoredLatency(Value val) {
      auto &node = valuesLatency[val];
      if (!node || node->pathLatency < 0) {
        if (isInputValue(val)) {
            TimingPathNodeOp* inputPath = new TimingPathNodeOp(0.0, val, nullptr);
            valuesLatency[val] = inputPath;
            return inputPath;
        }
      LLVM_DEBUG(llvm::dbgs()
                 << "could not find latency for Value " << val << " defined here: " << val.getDefiningOp() << "\n");
        return nullptr;
      }
      return node;
    }

    bool visitStmt(ConnectOp op) {
      Value srcOp;
      if (isWire(op.src())) srcOp = getWireValue(op.src());
      else srcOp = op.src();
      TimingPathNodeOp* path = getStoredLatency(srcOp);
      if (!path) return false;
      Value destOp;
      if (isWire(op.dest())) {
        // we squeeze wire as a single Value
        // todo: optimize by field
        destOp = getWireValue(op.dest());
      } else {
        destOp = op.dest();
      }
      TimingPathNodeOp *localPath = new TimingPathNodeOp(0.0, destOp, path);
      valuesLatency[destOp] = localPath;
      if (isOutputValue(destOp)) {
        LLVM_DEBUG(llvm::dbgs() << "found connection to output ");
        LLVM_DEBUG(destOp.print(llvm::dbgs()););
        LLVM_DEBUG(llvm::dbgs() << ".\n");
        outputPaths.push_back(localPath);
      } else if (isRegister(destOp)) {
        LLVM_DEBUG(llvm::dbgs() << "found connection to register ");
        LLVM_DEBUG(destOp.print(llvm::dbgs()););
        LLVM_DEBUG(llvm::dbgs() << ".\n");
        toRegPaths.push_back(localPath);
      }
      return true;
    }

    bool visitDecl(WireOp op){
      return true;
    }

    bool visitDecl(RegOp op){
      LLVM_DEBUG( llvm::dbgs() << "declaring register ";
        op.result().print(llvm::dbgs());
        llvm::dbgs() << ".\n");
      valuesLatency[op.result()] = new TimingPathNodeOp(0.0, op.result(), nullptr);
      return true;
    }

    bool visitDecl(InstanceOp op){
      LLVM_DEBUG(llvm::dbgs() << "visiting module instance\n");
      // lookup if module's critical paths have already been evaluated
      ModuleTimingInfo* moduleTimingInfo = moduleMap->getModuleTimingInfoByName(op.moduleName());
      if (!moduleTimingInfo) {
        llvm::errs() << "in InstanceOp, unknown module: " << op.moduleName() << "\n";
        return false;
      }
      // SmallVector<ModulePortInfo, 8> portInfo = getModulePortInfo(op);

      llvm::SmallDenseMap<StringRef, TimingPathNodeOp*> latencyByPortName;
      // processing input latencies first
      for (unsigned portIdx = 0, e = moduleTimingInfo->portInfo.size(); portIdx != e; ++portIdx) {
        auto port = moduleTimingInfo->portInfo[portIdx];
        if (port.isOutput()) continue;
        // input
        auto inputPath = getStoredLatency(op->getOpResult(portIdx));
        if (!inputPath) {
          llvm::outs() << "latency for input" << port.name << " of module " << op.moduleName() << " has not been determined yet.\n";
          return false;
        }
        latencyByPortName[port.getName()] = inputPath;
      }
      // processing outputs
      for (unsigned portIdx = 0, e = moduleTimingInfo->portInfo.size(); portIdx != e; ++portIdx) {
        auto port = moduleTimingInfo->portInfo[portIdx];
        if (!port.isOutput()) continue;

        StringRef portName = port.getName();
        ModulePathTimingInfo* path = moduleTimingInfo->getPathFromOutput(portName);
        if (path) { // todo: should check if port.getName() appears in map
          auto newPath = path->path.endPoint->copyPathUpstream();
          auto internalPathStart = newPath->getPathFirstNode(true /* discardCurrent*/ );
          // if the internal path start is a module input (not a register) we connect
          // its start to the path outside the instantiated module
          if (!isRegister(internalPathStart->nodeOp))
            internalPathStart->previousNode = latencyByPortName[path->startLabel];
          valuesLatency[op->getOpResult(portIdx)] = newPath->getPathLastNode();
        } else {
          llvm::errs() << "could not find internal path for output port " << portName << " of module " << op.moduleName() << ".\n";
          for (auto port : moduleTimingInfo->pathFromOutput) {
            llvm::errs() << "registered ports are: " << port.first << "\n";
          }
          llvm::report_fatal_error("module instancing failed.");
          return false;
        }
      }
      return true;
    }
};


/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CriticalPathAnalysisPass : public CriticalPathAnalysisBase<CriticalPathAnalysisPass> {
  // llvm::DenseMap<StringRef, ModuleTimingInfo> moduleTimings;

  void runOnOperation() override;

  MapModuleTimingInfo moduleMap;
};
} // namespace


void CriticalPathAnalysisPass::runOnOperation() {
  FModuleOp module = getOperation();

  // todo: try to pass displayLocation as a command-line option
  // applying CL options
  // circt::applyCriticalPathCLOptions(module);

  ModuleTimingInfo* moduleTiming = new ModuleTimingInfo(module);
  ModuleInfo moduleInfo;
  moduleInfo.module = module;

  llvm::outs() << "executing CriticalPathAnalysisPass on " << module.getName().str() << ".\n";

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Port " << port.getName() << " of type " << port.type << " isOutput " << port.isOutput() << "\n");
  }

  auto latencyEvaluator = ExprLatencyEvaluator(&moduleInfo, &moduleMap);

  // list of Operation node awaiting processing
  std::list<Operation*> worklist;

  // temporary determined critical path end node
  TimingPathNodeOp* criticalPathEnd = nullptr;

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
      auto& path = latencyEvaluator.valuesLatency[op->getOpResult(0)];
      if (!path) {
        if (nullptr == criticalPathEnd || criticalPathEnd->pathLatency < path->pathLatency)
        criticalPathEnd = path;
        LLVM_DEBUG(llvm::dbgs()
                  << "Found latency " << path->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      }
    }
  });


  // Depending on the order of operations during the initial list traversal
  // some operations timing may not have been determined, and we may need
  // to traverse the list of remaining (undetermined) nodes
  //
  // todo: check invariant in circt's FIRRTL operation order
  while (!worklist.empty()) {
    Operation* op = worklist.front();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing " << op->getName().getStringRef().str() << "\n");
    worklist.pop_front();

    if (discardableOp(op)) {
      LLVM_DEBUG(llvm::dbgs()
                << "Found to discard: " << op->getName().getStringRef().str() << "\n");
      continue;
    }

    bool latencyFound = latencyEvaluator.dispatchExprVisitor(op);

    if (latencyFound) {
      auto& path = latencyEvaluator.valuesLatency[op->getOpResult(0)];
      if (!path) {
        LLVM_DEBUG(llvm::dbgs()
                  << "invalid latency for op " << op->getName().getStringRef().str() << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                  << "Found latency " << path->pathLatency << " for op " << op->getName().getStringRef().str() << "\n");
      }
    } else {
      worklist.push_back(op);
    }
  }

  // todo: need a more canonical part to display result info
  // critical path traversal
  moduleTiming->registerOutputPath(latencyEvaluator.outputPaths);
  moduleTiming->registerPathToRegs(latencyEvaluator.toRegPaths);
  moduleTiming->displayPaths(true);
  LLVM_DEBUG(llvm::dbgs()
             << "register module timing info for " << module.getName() << "\n");
  moduleMap.registerModuleTimingInfo(module.getName(), moduleTiming);

  // register's module critical path

  // cleaning memory
  LLVM_DEBUG(llvm::dbgs() << "cleaning memory.\n");
  // todo: cleaning memory commented-out because we may need path in other modules
  // -> should be clean-up eventually
  // for (auto it : latencyEvaluator.valuesLatency) {
  //  delete it.second;
  // }

  // todo: keep ?
  markAllAnalysesPreserved();
}


std::unique_ptr<mlir::Pass> circt::firrtl::createCriticalPathAnalysisPass() {
  return std::make_unique<CriticalPathAnalysisPass>();
}

