//===- ClangFrontend.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangFrontend.hpp"
#include "frontend_impl.hpp"

#include "include/ClangFrontend.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <span>

namespace lcl {
namespace detail {
class ClangFrontendImpl {
public:
  ClangFrontendImpl() {
    mCompiler = std::make_unique<clang::CompilerInstance>();

    mDiagID = new clang::DiagnosticIDs();
    mDiagOpts = new clang::DiagnosticOptions();

    mDiagOpts->ShowPresumedLoc = true;
  }
  FrontendResult process(const std::string_view source,
                         const llvm::ArrayRef<llvm::StringRef> options) {
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

    llvm::SmallVector<std::string, 15> storage;
    llvm::SmallVector<const char *, 15> allOpts;
    for (llvm::StringRef opt : options) {
      storage.push_back(opt.str());
      auto &str = storage.back();
      allOpts.push_back(str.data());
    }
    allOpts.push_back("-x");
    allOpts.push_back("cl");
    allOpts.push_back("-emit-llvm-bc");
    allOpts.push_back("-triple");
    allOpts.push_back("spir64");
    allOpts.push_back("-fdeclare-opencl-builtins");
    allOpts.push_back("-disable-llvm-passes");
    allOpts.push_back("-cl-ext=all");
    allOpts.push_back("sample.cl");
    allOpts.push_back("-o");
    allOpts.push_back("-");
    allOpts.push_back("-cl-kernel-arg-info");

    clang::CompilerInvocation::CreateFromArgs(mCompiler->getInvocation(),
                                              allOpts, *diags);

    MemFS->addFile("sample.cl", (time_t)0,
                   llvm::MemoryBuffer::getMemBuffer(
                       llvm::StringRef(source.data()), "sample.cl"));

    bool success = clang::ExecuteCompilerInvocation(mCompiler.get());

    errStream.flush();
    if (!success) {
      return FrontendResult{errString};
    }

    llvm::StringRef irModule(static_cast<const char *>(irBuffer.data()),
                             irBuffer.size());

    std::unique_ptr<llvm::MemoryBuffer> MB =
        llvm::MemoryBuffer::getMemBuffer(irModule, "sample.bc", false);

    auto E = llvm::getOwningLazyBitcodeModule(std::move(MB), mContext,
                                              /*ShouldLazyLoadMetadata=*/
                                              true);
    if (auto err = E.takeError()) {
      errStream << toString(std::move(err)) << "\n";
      errStream.flush();
      return FrontendResult{errString};
    }

    std::unique_ptr<llvm::Module> M = std::move(*E);
    llvm::Error err = M->materializeAll();
    if (err) {
      errStream << toString(std::move(err)) << "\n";
      errStream.flush();
      return FrontendResult{errString};
    }

    return FrontendResult{std::make_shared<detail::Module>(std::move(M))};
  }

private:
  llvm::LLVMContext mContext;
  std::unique_ptr<clang::CompilerInstance> mCompiler;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> mDiagID;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> mDiagOpts;
};
} // namespace detail

ClangFrontend::ClangFrontend()
    : mImpl(std::make_shared<detail::ClangFrontendImpl>()) {}

FrontendResult ClangFrontend::process(std::string_view input,
                                      std::span<std::string_view> options) {
  llvm::SmallVector<llvm::StringRef, 15> opts;
  for (std::string_view opt : options) {
    opts.push_back(llvm::StringRef{opt});
  }
  return mImpl->process(input, opts);
}
} // namespace lcl
