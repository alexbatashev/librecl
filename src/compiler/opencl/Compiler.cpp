//===- Compiler.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.hpp"
#include "CppEmitter.hpp"
#include "LibreCL/IR/LibreCLDialect.h"
#include "Options.h"
#include "RawMemory/RawMemoryDialect.h"
#include "Struct/StructDialect.h"
#include "passes/llvm/FixupStructuredCFGPass.h"
#include "passes/mlir/passes.hpp"

#include "LLVMSPIRVLib.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdlib>
#include <exception>
#include <istream>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Operation.h>
#include <streambuf>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace lcl {

struct SourceInput {
  std::string_view source;
};

struct SPIRVInput {
  std::span<char> binary;
};

using JobInput = std::variant<SourceInput, SPIRVInput, CompileResult, std::span<CompileResult *>>;

class CompilerJob {
public:
  virtual CompileResult compile(const JobInput &input) = 0;
  virtual ~CompilerJob() = default;
};

class ClangFrontendJob : public CompilerJob {
public:
  ClangFrontendJob(llvm::LLVMContext &context, const ::Options &options)
      : mContext(context), mCompiler(nullptr) {
    mCompiler = std::make_unique<clang::CompilerInstance>();

    mDiagID = new clang::DiagnosticIDs();
    mDiagOpts = new clang::DiagnosticOptions();

    mDiagOpts->ShowPresumedLoc = true;

    for (size_t i = 0; i < options.num_other_options; i++) {
      mOptsStorage.emplace_back(options.other_options[i]);
      mOpts.push_back(mOptsStorage.back().c_str());
    }
    mOpts.push_back("-x");
    mOpts.push_back("cl");
    mOpts.push_back("-emit-llvm-bc");
    mOpts.push_back("-triple");
    mOpts.push_back("spir64");
    mOpts.push_back("-fdeclare-opencl-builtins");
    mOpts.push_back("-disable-llvm-passes");
    mOpts.push_back("-cl-ext=all");
    mOpts.push_back("source.cl");
    mOpts.push_back("-o");
    mOpts.push_back("-");
    mOpts.push_back("-cl-kernel-arg-info");
  }

  CompileResult compile(const JobInput &input) final {
    if (!std::holds_alternative<SourceInput>(input)) {
      return CompileResult("invalid clang input");
    }

    auto source = std::get<SourceInput>(input).source;

    std::string errString;
    llvm::raw_string_ostream errStream(errString);

    // TODO use unique_ptr
    clang::TextDiagnosticPrinter *diagsPrinter =
        new clang::TextDiagnosticPrinter(errStream, &*mDiagOpts);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
        new clang::DiagnosticsEngine(mDiagID, &*mDiagOpts, diagsPrinter));

    llvm::SmallVector<char, 4096> irBuffer;

    auto irStream = std::make_unique<llvm::raw_svector_ostream>(irBuffer);

    mCompiler->setOutputStream(std::move(irStream));
    mCompiler->setDiagnostics(&*diags);

    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS(
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> MemFS(
        new llvm::vfs::InMemoryFileSystem);
    OverlayFS->pushOverlay(MemFS);

    mCompiler->createFileManager(OverlayFS);
    mCompiler->createSourceManager(mCompiler->getFileManager());

    llvm::cl::ResetAllOptionOccurrences();

    clang::CompilerInvocation::CreateFromArgs(mCompiler->getInvocation(), mOpts,
                                              *diags);

    MemFS->addFile("source.cl", (time_t)0,
                   llvm::MemoryBuffer::getMemBuffer(
                       llvm::StringRef(source.data()), "source.cl"));

    bool success = clang::ExecuteCompilerInvocation(mCompiler.get());

    errStream.flush();
    if (!success) {
      return CompileResult{errString};
    }

    llvm::StringRef irModule(static_cast<const char *>(irBuffer.data()),
                             irBuffer.size());

    std::unique_ptr<llvm::MemoryBuffer> MB =
        llvm::MemoryBuffer::getMemBuffer(irModule, "source.bc", false);

    auto E = llvm::getOwningLazyBitcodeModule(std::move(MB), mContext,
                                              /*ShouldLazyLoadMetadata=*/
                                              true);
    if (auto err = E.takeError()) {
      errStream << toString(std::move(err)) << "\n";
      errStream.flush();
      return CompileResult{errString};
    }

    std::unique_ptr<llvm::Module> M = std::move(*E);
    llvm::Error err = M->materializeAll();
    if (err) {
      errStream << toString(std::move(err)) << "\n";
      errStream.flush();
      return CompileResult{errString};
    }

    return CompileResult{std::move(M)};
  }

  ~ClangFrontendJob() override = default;

private:
  llvm::LLVMContext &mContext;
  std::unique_ptr<clang::CompilerInstance> mCompiler;
  llvm::SmallVector<std::string, 10> mOptsStorage;
  llvm::SmallVector<const char *, 10> mOpts;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> mDiagID;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> mDiagOpts;
};

class LLVMIROptimizeJob : public CompilerJob {
public:
  LLVMIROptimizeJob(const ::Options &options) {
    using namespace llvm;

    PassBuilder PB;

    PB.registerModuleAnalyses(mMAM);
    PB.registerCGSCCAnalyses(mCGAM);
    PB.registerFunctionAnalyses(mFAM);
    PB.registerLoopAnalyses(mLAM);
    PB.crossRegisterProxies(mLAM, mFAM, mCGAM, mMAM);

    // TODO right now compiler requires O2 pipeline in order to be able to
    // consume LLVM IR. This should be something like O0 if -cl-opt-none is
    // passed.
    mPassManager = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
    mPassManager.addPass(
        createModuleToFunctionPassAdaptor(clspv::FixupStructuredCFGPass()));
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<CompileResult>(input)) {
      return CompileResult{"invalid LLVM optimizer input"};
    }

    auto &result = std::get<CompileResult>(input);
    if (result.isError())
      return CompileResult{result.getError()};
    if (!result.hasLLVMIR()) {
      return CompileResult{"LLVMIROptimizeJob: not an LLVM IR module"};
    }

    auto clone = llvm::CloneModule(*result.getLLVMIR());
    mPassManager.run(*clone, mMAM);

    return CompileResult{std::move(clone)};
  }

  ~LLVMIROptimizeJob() override = default;

private:
  llvm::LoopAnalysisManager mLAM;
  llvm::FunctionAnalysisManager mFAM;
  llvm::CGSCCAnalysisManager mCGAM;
  llvm::ModuleAnalysisManager mMAM;
  llvm::ModulePassManager mPassManager;
};

class ConvertToMLIRJob : public CompilerJob {
public:
  ConvertToMLIRJob(mlir::MLIRContext *context, const ::Options &options)
      : mContext(context), mPassManager(context) {
    using namespace mlir;

    const bool debugMLIR = std::getenv("LIBRECL_DEBUG_MLIR") != nullptr;
    bool printBeforeAll = debugMLIR || options.print_before_mlir;
    bool printAfterAll = debugMLIR || options.print_after_mlir;

    if (printBeforeAll || printAfterAll) {
      const auto shouldPrintBeforeAll = [printBeforeAll](mlir::Pass *,
                                                         mlir::Operation *) {
        return printBeforeAll;
      };
      const auto shouldPrintAfterAll = [printAfterAll](mlir::Pass *,
                                                       mlir::Operation *) {
        return printAfterAll;
      };
      mContext->disableMultithreading();
      mPassManager.enableIRPrinting(shouldPrintBeforeAll, shouldPrintAfterAll);
    }
    if (debugMLIR) {
      mPassManager.enableTiming();
      mPassManager.enableStatistics();
    }

    mPassManager.addPass(mlir::createCanonicalizerPass());
    mPassManager.addPass(createSPIRToGPUPass());
    mPassManager.addPass(mlir::createCanonicalizerPass());
    mPassManager.addPass(lcl::createInferPointerTypesPass());
    mPassManager.addPass(mlir::createCanonicalizerPass());
    mPassManager.addNestedPass<mlir::gpu::GPUModuleOp>(
        createStructureCFGPass());
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<CompileResult>(input)) {
      return CompileResult{"invalid MLIR conversion input"};
    }

    auto &result = std::get<CompileResult>(input);
    if (result.isError())
      return CompileResult{result.getError()};

    if (!result.hasLLVMIR()) {
      return CompileResult{"ConvertToMLIRJob: not an LLVM IR module"};
    }

    auto clone = llvm::CloneModule(*result.getLLVMIR());
    auto mlirModule = mlir::translateLLVMIRToModule(std::move(clone), mContext);

    auto passResult = mPassManager.run(mlirModule.get());

    if (mlir::failed(passResult)) {
      return CompileResult{"failed to convert LLVM to MLIR"};
    }

    return CompileResult{std::move(mlirModule)};
  }

  ~ConvertToMLIRJob() = default;

private:
  mlir::MLIRContext *mContext;
  mlir::PassManager mPassManager;
};

class MergeMLIRJob : public CompilerJob {
public:
  MergeMLIRJob(mlir::MLIRContext *context, const ::Options &options)
      : mContext(context), mBuilder(context) {
    using namespace mlir;
    registerAllDialects(*mContext);
    DialectRegistry registry;
    registry.insert<rawmem::RawMemoryDialect>();
    registry.insert<structure::StructDialect>();
    registry.insert<lcl::LibreCLDialect>();
    mContext->appendDialectRegistry(registry);
    mContext->loadAllAvailableDialects();
    registerAllPasses();
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<std::span<CompileResult*>>(input)) {
      return CompileResult{"invalid merge job input"};
    }

    const bool debugMLIR = std::getenv("LIBRECL_DEBUG_MLIR") != nullptr;

    auto mlirModule = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mBuilder.getUnknownLoc()));
    mBuilder.setInsertionPointToStart(mlirModule->getBody());
    auto gpuModule = mBuilder.create<mlir::gpu::GPUModuleOp>(mBuilder.getUnknownLoc(), "ocl_program");

    const auto &modules = std::get<std::span<CompileResult *>>(input);

    for (const auto &res : modules) {
      if (!res) {
        return CompileResult{"one of the modules being linked is invalid"};
      }
      if (res->isError()) {
        return CompileResult{res->getError()};
      }
      const auto &m = res->getMLIR();

      if (m.get()->hasAttr(mlir::spirv::getTargetEnvAttrName())) {
        if (!mlirModule->getOperation()->hasAttr(
                mlir::spirv::getTargetEnvAttrName())) {
          mlirModule->getOperation()->setAttr(
              mlir::spirv::getTargetEnvAttrName(),
              m.get()->getAttr(mlir::spirv::getTargetEnvAttrName()));
        }
      }

      mlir::BlockAndValueMapping mapping;
      m.get()->walk([&](mlir::gpu::GPUModuleOp gm) {
        mlir::OpBuilder::InsertionGuard _{mBuilder};
        mBuilder.setInsertionPoint(gpuModule.body().front().getTerminator());
        for (auto &op : gm.body().front()) {
          bool skip = llvm::isa<mlir::gpu::ModuleEndOp>(op);

          if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            if (func.isDeclaration() &&
                gpuModule.lookupSymbol(func.getName())) {
              skip = true;
            }
            if (!func.isDeclaration()) {
              auto otherFunc =
                  gpuModule.lookupSymbol<mlir::func::FuncOp>(func.getName());
              if (otherFunc) {
                if (otherFunc.isDeclaration()) {
                  otherFunc.erase();
                } else {
                  // TODO how to return errors?
                  return; // CompileResult{"can not have multiple functions with
                          // the same name"};
                }
              }
            }
          }
          /*
          if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            if (gpuModule.lookupSymbol(func.getSymName())
          }
          */
          if (!skip)
            mBuilder.clone(op, mapping);
        }
      });
      // TODO currently LibreCL compiler emits only device code. The plan is to
      // produce a fat binary with WebAssembly and device-native parts. This
      // would require cloning functions and other metadata from outside of gpu
      // module region.
    }

    if (debugMLIR) {
      mlirModule->dump();
    }

    return CompileResult{std::move(mlirModule)};
  }

  ~MergeMLIRJob() = default;

private:
  mlir::MLIRContext *mContext;
  mlir::OpBuilder mBuilder;
};

class ConvertMLIRToMSLJob : public CompilerJob {
public:
  ConvertMLIRToMSLJob(mlir::MLIRContext *context, const ::Options &options)
      : mContext(context), mPassManager(context) {
    using namespace mlir;
    registerAllDialects(*mContext);
    DialectRegistry registry;
    registry.insert<rawmem::RawMemoryDialect>();
    registry.insert<structure::StructDialect>();
    mContext->appendDialectRegistry(registry);
    mContext->loadAllAvailableDialects();
    registerAllPasses();

    const bool debugMLIR = std::getenv("LIBRECL_DEBUG_MLIR") != nullptr;
    bool printBeforeAll = debugMLIR || options.print_before_mlir;
    bool printAfterAll = debugMLIR || options.print_after_mlir;

    if (printBeforeAll || printAfterAll) {
      const auto shouldPrintBeforeAll = [printBeforeAll](mlir::Pass *,
                                                         mlir::Operation *) {
        return printBeforeAll;
      };
      const auto shouldPrintAfterAll = [printAfterAll](mlir::Pass *,
                                                       mlir::Operation *) {
        return printAfterAll;
      };
      mContext->disableMultithreading();
      mPassManager.enableIRPrinting(shouldPrintBeforeAll, shouldPrintAfterAll);
    }
    if (debugMLIR) {
      mPassManager.enableTiming();
      mPassManager.enableStatistics();
    }

    mPassManager.addNestedPass<mlir::gpu::GPUModuleOp>(
        createAIRKernelABIPass());
    mPassManager.addPass(createExpandGPUBuiltinsPass());
    mPassManager.addPass(mlir::createCanonicalizerPass());
    mPassManager.addPass(createGPUToCppPass());
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<CompileResult>(input)) {
      return CompileResult{"invalid MLIR input"};
    }

    auto &result = std::get<CompileResult>(input);
    if (result.isError())
      return CompileResult{result.getError()};
    if (!result.hasMLIR()) {
      return CompileResult{"not an MLIR module"};
    }

    auto mlirModule = mlir::OwningOpRef<mlir::ModuleOp>(
        const_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(result.getMLIR())
            ->clone());

    std::vector<KernelInfo> kernels;

    // TODO account for data layout types
    const auto getTypeSize = [](mlir::Type type) -> size_t {
      if (type.isa<mlir::rawmem::PointerType>()) {
        return 8;
      }
      if (type.isIntOrFloat()) {
        return type.getIntOrFloatBitWidth() / 8;
      }

      return -1;
    };

    mlirModule->walk([&kernels, &getTypeSize](mlir::gpu::GPUFuncOp func) {
      if (!func.isKernel())
        return;
      std::string name = func.getName().str();
      std::vector<ArgumentInfo> args;

      for (auto arg : func.getArgumentTypes()) {
        size_t size = getTypeSize(arg);
        if (arg.isa<mlir::rawmem::PointerType>()) {
          args.push_back(
              ArgumentInfo{.type = ArgumentInfo::ArgType::GlobalBuffer,
                           .index = args.size(),
                           .size = size});
        } else {
          args.push_back(ArgumentInfo{.type = ArgumentInfo::ArgType::POD,
                                      .index = args.size(),
                                      .size = size});
        }
      }

      kernels.emplace_back(name, args);
    });

    auto passResult = mPassManager.run(*mlirModule);

    if (mlir::failed(passResult)) {
      return CompileResult{"failed to lower MLIR for MSL conversion"};
    }

    std::string source;
    {
      llvm::raw_string_ostream mslStream{source};
      // TODO check for errors
      auto result = lcl::translateToCpp(mlirModule.get(), mslStream, false);
      if (mlir::failed(result)) {
        return CompileResult{"Failed to convert MLIR to MSL"};
      }

      mslStream.flush();
    }

    std::vector<unsigned char> resBinary{
        reinterpret_cast<unsigned char *>(source.data()),
        reinterpret_cast<unsigned char *>(source.data() + source.size())};

    return CompileResult{BinaryProgram{resBinary, kernels}};
  }

private:
  mlir::MLIRContext *mContext;
  mlir::PassManager mPassManager;
};
class ConvertMLIRToSPIRVJob : public CompilerJob {
public:
  ConvertMLIRToSPIRVJob(mlir::MLIRContext *context, const ::Options &options)
      : mContext(context), mPassManager(context) {
    using namespace mlir;
    registerAllDialects(*mContext);
    DialectRegistry registry;
    registry.insert<rawmem::RawMemoryDialect>();
    registry.insert<structure::StructDialect>();
    mContext->appendDialectRegistry(registry);
    mContext->loadAllAvailableDialects();
    registerAllPasses();

    const bool debugMLIR = std::getenv("LIBRECL_DEBUG_MLIR") != nullptr;
    bool printBeforeAll = debugMLIR || options.print_before_mlir;
    bool printAfterAll = debugMLIR || options.print_after_mlir;

    if (printBeforeAll || printAfterAll) {
      const auto shouldPrintBeforeAll = [printBeforeAll](mlir::Pass *,
                                                         mlir::Operation *) {
        return printBeforeAll;
      };
      const auto shouldPrintAfterAll = [printAfterAll](mlir::Pass *,
                                                       mlir::Operation *) {
        return printAfterAll;
      };
      mContext->disableMultithreading();
      mPassManager.enableIRPrinting(shouldPrintBeforeAll, shouldPrintAfterAll);
    }
    if (debugMLIR) {
      mPassManager.enableTiming();
      mPassManager.enableStatistics();
    }

    mPassManager.addPass(lcl::createGPUToSPIRVPass());
    mPassManager.addNestedPass<mlir::spirv::ModuleOp>(
        lcl::createLowerABIAttributesPass());
    mPassManager.addNestedPass<mlir::spirv::ModuleOp>(
        mlir::spirv::createUpdateVersionCapabilityExtensionPass());
    // TODO should not be necessary
    if (options.opt_level >= 2) {
      mPassManager.addNestedPass<mlir::spirv::ModuleOp>(
          mlir::createInlinerPass());
      mPassManager.addNestedPass<mlir::spirv::ModuleOp>(mlir::createCSEPass());
    }
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<CompileResult>(input)) {
      return CompileResult{"invalid MLIR optimizer input"};
    }

    auto &result = std::get<CompileResult>(input);
    if (result.isError())
      return CompileResult{result.getError()};
    if (!result.hasMLIR()) {
      return CompileResult{"not an MLIR module"};
    }

    auto mlirModule = mlir::OwningOpRef<mlir::ModuleOp>(
        const_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(result.getMLIR())
            ->clone());

    auto passResult = mPassManager.run(*mlirModule);

    if (mlir::failed(passResult)) {
      return CompileResult{"failed to convert MLIR to SPIR-V"};
    }

    llvm::SmallVector<uint32_t, 10000> binary;

    auto spvModule =
        mlirModule->lookupSymbol<mlir::spirv::ModuleOp>("__spv__ocl_program");
    if (spvModule) {
      auto conversionResult = mlir::spirv::serialize(spvModule, binary);
      if (mlir::failed(conversionResult)) {
        return CompileResult{"Failed to serialize SPIR-V"};
      }
    } else {
      return CompileResult{"Failed to locate SPIR-V module"};
    }

    std::vector<unsigned char> resBinary;
    resBinary.resize(sizeof(uint32_t) * binary.size());
    std::memcpy(resBinary.data(), binary.data(), resBinary.size());

    std::vector<KernelInfo> kernels;

    // TODO account for data layout types
    const auto getTypeSize = [](mlir::Type type) -> size_t {
      if (type.isa<mlir::rawmem::PointerType>()) {
        return 8;
      }
      if (type.isIntOrFloat()) {
        return type.getIntOrFloatBitWidth() / 8;
      }

      return -1;
    };

    mlirModule->walk([&kernels, &getTypeSize](mlir::gpu::GPUFuncOp func) {
      if (!func.isKernel())
        return;
      std::string name = func.getName().str();
      std::vector<ArgumentInfo> args;

      for (auto arg : func.getArgumentTypes()) {
        size_t size = getTypeSize(arg);
        if (arg.isa<mlir::rawmem::PointerType>()) {
          args.push_back(
              ArgumentInfo{.type = ArgumentInfo::ArgType::GlobalBuffer,
                           .index = args.size(),
                           .size = size});
        } else {
          args.push_back(ArgumentInfo{.type = ArgumentInfo::ArgType::POD,
                                      .index = args.size(),
                                      .size = size});
        }
      }

      kernels.emplace_back(name, args);
    });

    return CompileResult{BinaryProgram{resBinary, kernels}};
  }

private:
  mlir::MLIRContext *mContext;
  mlir::PassManager mPassManager;
};

class MLIRLTOJob : public CompilerJob {
public:
  MLIRLTOJob(mlir::MLIRContext *context, const ::Options &options)
      : mContext(context), mPassManager(context) {
    using namespace mlir;
    registerAllDialects(*mContext);
    DialectRegistry registry;
    registry.insert<rawmem::RawMemoryDialect>();
    registry.insert<structure::StructDialect>();
    mContext->appendDialectRegistry(registry);
    mContext->loadAllAvailableDialects();
    registerAllPasses();

    const bool debugMLIR = std::getenv("LIBRECL_DEBUG_MLIR") != nullptr;
    bool printBeforeAll = debugMLIR || options.print_before_mlir;
    bool printAfterAll = debugMLIR || options.print_after_mlir;

    if (printBeforeAll || printAfterAll) {
      const auto shouldPrintBeforeAll = [printBeforeAll](mlir::Pass *,
                                                         mlir::Operation *) {
        return printBeforeAll;
      };
      const auto shouldPrintAfterAll = [printAfterAll](mlir::Pass *,
                                                       mlir::Operation *) {
        return printAfterAll;
      };
      mContext->disableMultithreading();
      mPassManager.enableIRPrinting(shouldPrintBeforeAll, shouldPrintAfterAll);
    }
    if (debugMLIR) {
      mPassManager.enableTiming();
      mPassManager.enableStatistics();
    }

    mPassManager.addPass(mlir::createCanonicalizerPass());
    mPassManager.addPass(mlir::createSymbolDCEPass());
    if (options.opt_level >= 2) {
      llvm::StringMap<mlir::OpPassManager> pipelines;
      const auto createPipeline = [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      };
      mPassManager.addNestedPass<mlir::gpu::GPUModuleOp>(
          mlir::createInlinerPass(pipelines, createPipeline));
    }
  }

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<CompileResult>(input)) {
      return CompileResult{"MLIRLTOJob: invalid input"};
    }
    const auto &result = std::get<CompileResult>(input);
    if (result.isError()) {
      return CompileResult{result.getError()};
    }
    if (!result.hasMLIR()) {
      return CompileResult{"MLIRLTOJob: input does not contain MLIR module"};
    }

    auto mlirModule = mlir::OwningOpRef<mlir::ModuleOp>(
        const_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(result.getMLIR())
            ->clone());

    auto passResult = mPassManager.run(mlirModule.get());

    if (mlir::failed(passResult)) {
      return CompileResult{"failed to optimize MLIR module"};
    }

    return CompileResult{std::move(mlirModule)};
  }

  ~MLIRLTOJob() = default;

private:
  mlir::MLIRContext *mContext;
  mlir::PassManager mPassManager;
};

template <typename DataT, typename TraitsT = std::char_traits<DataT>>
class spanbuffer : public std::basic_streambuf<DataT, TraitsT> {
  using Base = std::basic_streambuf<DataT, TraitsT>;

public:
  spanbuffer(std::span<DataT> &span) {
    Base::setg(span.data(), span.data(), span.data() + span.size());
  }
};

class SPIRVTranslatorJob : public CompilerJob {
public:
  // TODO support specialization constants
  SPIRVTranslatorJob(llvm::LLVMContext &context) : mContext(context) {}

  CompileResult compile(const JobInput &input) override {
    if (!std::holds_alternative<SPIRVInput>(input)) {
      return CompileResult{"Not a valid SPIR-V module"};
    }

    std::span<char> spirv = std::get<SPIRVInput>(input).binary;

    std::string errMessage;

    spanbuffer buffer{spirv};
    std::istream stream{&buffer};
    llvm::Module *modulePtr;
    if (!llvm::readSpirv(mContext, stream, modulePtr, errMessage)) {
      return CompileResult{errMessage};
    }

    if (!modulePtr) {
      return CompileResult{errMessage};
    }

    std::unique_ptr<llvm::Module> module{modulePtr};
    return CompileResult{std::move(module)};
  }

  ~SPIRVTranslatorJob() override = default;

private:
  llvm::LLVMContext &mContext;
};

static CompileResult do_compile(llvm::SmallVectorImpl<std::unique_ptr<CompilerJob>> &jobs, JobInput &input) {
    for (auto &job : jobs) {
      input = job->compile(input);
    }

    if (std::holds_alternative<CompileResult>(input)) {
      return std::move(std::get<CompileResult>(input));
    } else {
      return CompileResult{"Failed to produce correct result"};
    }
}

Compiler::Compiler() {
  using namespace mlir;
  registerAllDialects(mMLIRContext);
  DialectRegistry registry;
  registry.insert<rawmem::RawMemoryDialect>();
  registry.insert<structure::StructDialect>();
  mMLIRContext.appendDialectRegistry(registry);
  mMLIRContext.loadAllAvailableDialects();
  registerAllPasses();
}

CompileResult Compiler::compile_from_mlir(std::span<const char> source,
                                const ::Options &options) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef{source.data(), source.size()}, &mMLIRContext);
  if (std::getenv("LIBRECL_DEBUG_MLIR") != nullptr) {
    module->dump();
  }
  return CompileResult{std::move(module)};
}

CompileResult Compiler::compile(std::span<const char> source,
                                const ::Options &options) {
  llvm::SmallVector<std::unique_ptr<CompilerJob>, 10> jobs;

  auto input = [&]() {
    if (source.size() % sizeof(uint32_t) == 0) {
      auto maybeSpirv = std::span<const uint32_t>(
          reinterpret_cast<const uint32_t *>(source.data()),
          reinterpret_cast<const uint32_t *>(source.data()) +
              source.size() / sizeof(uint32_t));
      if (maybeSpirv[0] == 0x07230203u) {
        return JobInput{SPIRVInput{std::span<char>{
            const_cast<char *>(source.data()),
            const_cast<char *>(source.data()) + source.size()}}};
      }
    }

    return JobInput{
        SourceInput{std::string_view{source.data(), source.size()}}};
  }();

  if (std::holds_alternative<SourceInput>(input)) {
    jobs.push_back(createStandardClangJob(options));
  } else if (std::holds_alternative<SPIRVInput>(input)) {
    jobs.push_back(createSPIRVTranslatorJob(options));
  } else {
    return CompileResult{"Failed to choose frontend"};
  }
  jobs.push_back(createOptimizeLLVMIRJob(options));
  jobs.push_back(createConvertToMLIRJob(options));

  return do_compile(jobs, input);
}

CompileResult Compiler::link(std::span<CompileResult *> modules,
                             const ::Options &options) {
  if (modules.size() == 0) {
    return CompileResult{"There must be at least 1 module for linking"};
  }

  JobInput input{modules};

  llvm::SmallVector<std::unique_ptr<CompilerJob>, 10> jobs;

  jobs.push_back(createMergeMLIRJob(options));

  if (options.opt_level > 1) {
    jobs.push_back(createMLIRLTOJob(options));
  }

  if (options.target_vulkan_spv) {
    jobs.push_back(createConvertMLIRToVulkanSPIRVJob(options));
  }

  if (options.target_metal_macos || options.target_metal_ios) {
    jobs.push_back(createConvertMLIRToMSLJob(options));
  }

  return do_compile(jobs, input);
}

std::unique_ptr<CompilerJob>
Compiler::createStandardClangJob(const ::Options &options) {
  auto ptr = std::make_unique<ClangFrontendJob>(mLLVMContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createOptimizeLLVMIRJob(const ::Options &options) {
  auto ptr = std::make_unique<LLVMIROptimizeJob>(options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createMergeMLIRJob(const ::Options &options) {
  auto ptr = std::make_unique<MergeMLIRJob>(&mMLIRContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createMLIRLTOJob(const ::Options &options) {
  auto ptr = std::make_unique<MLIRLTOJob>(&mMLIRContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createConvertToMLIRJob(const ::Options &options) {
  auto ptr = std::make_unique<ConvertToMLIRJob>(&mMLIRContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createConvertMLIRToVulkanSPIRVJob(const ::Options &options) {
  auto ptr = std::make_unique<ConvertMLIRToSPIRVJob>(&mMLIRContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createConvertMLIRToMSLJob(const ::Options &options) {
  auto ptr = std::make_unique<ConvertMLIRToMSLJob>(&mMLIRContext, options);
  return ptr;
}

std::unique_ptr<CompilerJob>
Compiler::createSPIRVTranslatorJob(const ::Options &options) {
  auto ptr = std::make_unique<SPIRVTranslatorJob>(mLLVMContext);
  return ptr;
}

} // namespace lcl
