//===- CriticalPathOptions.h - CIRCT Lowering Options ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process and verilog exporting.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_CRITICALPATHOPTIONS_H
#define CIRCT_SUPPORT_CRITICALPATHOPTIONS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;
}

namespace circt {

/// Options which control the emission from CIRCT to Verilog.
struct CriticalPathOptions {
  /// Error callback type used to indicate errors parsing the options string.
  using ErrorHandlerT = function_ref<void(llvm::Twine)>;

  /// Create a CriticalPathOptions with the default values.
  CriticalPathOptions() = default;

  /// Create a CriticalPathOptions and read in options from a string,
  /// overriding only the set options in the string.
  CriticalPathOptions(StringRef options, ErrorHandlerT errorHandler);

  /// Create a CriticalPathOptions with values loaded from an MLIR ModuleOp. This
  /// loads a string attribute with the key `circt.loweringOptions`. If there is
  /// an error parsing the attribute this will print an error using the
  /// ModuleOp.
  CriticalPathOptions(mlir::ModuleOp module);

  /// Read in options from a string, overriding only the set options in the
  /// string.
  void parse(StringRef options, ErrorHandlerT callback);

  /// Returns a string representation of the options.
  std::string toString() const;

  /// Write the verilog emitter options to a module's attributes.
  void setAsAttribute(mlir::ModuleOp module);

  // Load any emitter options from the module. If there is an error validating
  // the attribute, this will print an error using the ModuleOp.
  void parseFromAttribute(mlir::ModuleOp module);

  // If true display code source location alongside critical path nodes
  bool displayLocation = true;

};

/// Register commandline options for the verilog emitter.
void registerCriticalPathCLOptions();

/// Apply any command line specified style options to the mlir module.
void applyCriticalPathCLOptions(ModuleOp module);

} // namespace circt

#endif // CIRCT_SUPPORT_CRITICALPATHOPTIONS_H