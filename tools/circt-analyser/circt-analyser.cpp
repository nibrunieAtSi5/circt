//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Transforms/Passes.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

/// Allow the user to specify the input file format.  This can be used to
/// override the input, and can be used to specify ambiguous cases like standard
/// input.
enum InputFormatKind { InputUnspecified, InputFIRFile, InputMLIRFile };


static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
    outputFilename("o",
                   cl::desc("Output filename, or directory for split output"),
                   cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"));

static cl::opt<bool> inliner("inline",
                             cl::desc("Run the FIRRTL module inliner"),
                             cl::init(false));


/// Process a single buffer of the input.
static LogicalResult
processBuffer(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              TimingScope &ts,
              MLIRContext &context) {
  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                      sv::SVDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Nothing in the parser is threaded.  Disable synchronization overhead.
  auto isMultithreaded = context.isMultithreadingEnabled();
  context.disableMultithreading();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  OwningModuleRef module;

  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile(sourceMgr, &context);

  if (!module)
    return failure();

  // Allow optimizations to run multithreaded.
  // context.enableMultithreading(isMultithreaded);


   // enable timing analysis
   auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
   modulePM.addPass(firrtl::createCriticalPathAnalysisPass());

  if (failed(pm.run(module.get())))
    return failure();

  auto outputTimer = ts.nest("Output");

  // Note that we intentionally "leak" the Module into the MLIRContext instead
  // of deallocating it.  There is no need to deallocate it right before
  // process exit.
  return success(); // callback(module.release());
}

/// This implements the top-level logic for the firtool command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeAnalyser(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }


  auto result = processBuffer(std::move(input), ts, context);
  if (failed(result))
    return failure();

  return success();
}

/// Main driver for firtool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeFirtool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "circt modular optimizer driver\n");
  MLIRContext context;

  // Do the guts of the firtool process.
  auto result = executeAnalyser(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
