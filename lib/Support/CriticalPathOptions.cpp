//===- CriticalPathOptions.cpp - CIRCT Lowering Options -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process. Contains command line
// option definitions and support.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CriticalPathOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// CriticalPathOptions
//===----------------------------------------------------------------------===//

CriticalPathOptions::CriticalPathOptions(StringRef options, ErrorHandlerT errorHandler)
    : CriticalPathOptions() {
  parse(options, errorHandler);
}

CriticalPathOptions::CriticalPathOptions(mlir::ModuleOp module) : CriticalPathOptions() {
  parseFromAttribute(module);
}

void CriticalPathOptions::parse(StringRef text, ErrorHandlerT errorHandler) {
  while (!text.empty()) {
    // Remove the first option from the text.
    auto split = text.split(",");
    auto option = split.first.trim();
    text = split.second;
    if (option == "") {
      // Empty options are fine.
    } else if (option == "displayloc") {
      displayLocation = true;
    } else {
      errorHandler(llvm::Twine("unknown style option \'") + option + "\'");
      // We continue parsing options after a failure.
    }
  }
}

std::string CriticalPathOptions::toString() const {
  std::string options = "";
  // All options should add a trailing comma to simplify the code.
  if (!displayLocation)
    options += "displayLocation,";

  // Remove a trailing comma if present.
  if (!options.empty()) {
    assert(options.back() == ',' && "all options should add a trailing comma");
    options.pop_back();
  }
  return options;
}

void CriticalPathOptions::setAsAttribute(ModuleOp module) {
  module->setAttr("circt.criticalPathOptions",
                  StringAttr::get(module.getContext(), toString()));
}

void CriticalPathOptions::parseFromAttribute(ModuleOp module) {
  if (auto styleAttr =
          module->getAttrOfType<StringAttr>("circt.criticalPathOptions")) {
    parse(styleAttr.getValue(), [&](Twine error) { module.emitError(error); });
  }
}

//===----------------------------------------------------------------------===//
// Command Line Option Processing
//===----------------------------------------------------------------------===//

namespace {
/// Commandline parser for CriticalPathOptions.  Delegates to the parser
/// defined by CriticalPathOptions.
struct CriticalPathOptionParser : public llvm::cl::parser<CriticalPathOptions> {

  CriticalPathOptionParser(llvm::cl::Option &option)
      : llvm::cl::parser<CriticalPathOptions>(option) {}

  bool parse(llvm::cl::Option &option, StringRef argName, StringRef argValue,
             CriticalPathOptions &value) {
    bool failed = false;
    value.parse(argValue, [&](Twine error) { failed = option.error(error); });
    return failed;
  }
};

/// Commandline arguments for critical path analysis
/// the command line arguments in multiple tools.
struct CriticalPathCLOptions {
  llvm::cl::opt<CriticalPathOptions, false, CriticalPathOptionParser> criticalPathOptions{
      "critical-path-options", llvm::cl::desc("Style options"),
      llvm::cl::value_desc("option")};
};
} // namespace

/// The staticly initialized command line options.
static llvm::ManagedStatic<CriticalPathCLOptions> clOptions;

void circt::registerCriticalPathCLOptions() { *clOptions; }

void circt::applyCriticalPathCLOptions(ModuleOp module) {
  // If the command line options were not registered in the first place, there
  // is nothing to parse.
  if (!clOptions.isConstructed())
    return;

  // If an output style is applied on the command line, all previous options are
  // discarded.
  if (clOptions->criticalPathOptions.getNumOccurrences()) {
    clOptions->criticalPathOptions.setAsAttribute(module);
  }
}