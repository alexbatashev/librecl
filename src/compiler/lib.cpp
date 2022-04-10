#include <string_view>
#include <iostream>

#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "clang/FrontendTool/Utils.h"

#include "compiler.hpp"
#include "LCLJIT.hpp"

namespace lcl {
void initializeTargets() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

class CompilerImpl {
 public:
  CompilerImpl(BuildTarget target) {
    mCompiler = std::make_unique<clang::CompilerInstance>();

    mDiagID = new clang::DiagnosticIDs();
    mDiagOpts = new clang::DiagnosticOptions();

    mDiagOpts->ShowPresumedLoc = true;

    auto JIT = LCLJIT::create();
    if (JIT)
      mJIT = std::move(JIT.get());
  }

  void addModuleFromSource(const std::string_view source, const std::string_view options) {
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

    clang::CompilerInvocation::CreateFromArgs(mCompiler->getInvocation(),
        {"-cl-std=CL3.0", "-emit-llvm-bc", "-fdeclare-opencl-builtins",
        "sample.cl", "-o", "-", "-disable-llvm-passes", "-cl-ext=all",
        "-cl-kernel-arg-info", "-triple", "nvptx64-unknown-unknown"},
      *diags);

    MemFS->addFile(
      "sample.cl", (time_t)0,
      llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(source.data()), "sample.cl"));

    bool success = clang::ExecuteCompilerInvocation(mCompiler.get());

    errStream.flush();

    llvm::StringRef irModule(static_cast<const char*>(irBuffer.data()),
          irBuffer.size());

    std::unique_ptr<llvm::MemoryBuffer> MB = llvm::MemoryBuffer::getMemBuffer(irModule, "sample.bc", false);

    llvm::LLVMContext Context;
    auto E = llvm::getOwningLazyBitcodeModule(std::move(MB), Context,
        /*ShouldLazyLoadMetadata=*/
        true);
    llvm::logAllUnhandledErrors(E.takeError(), errStream, "error: ");
    std::unique_ptr<llvm::Module> M = std::move(*E);

    errStream.flush();

    if (M) {
      llvm::Error err = M->materializeAll();
      if (!err) {
        M->print(llvm::outs(), nullptr);
      }
    }
    else
      std::cout << errString << "\n";
  }
private:
  std::unique_ptr<clang::CompilerInstance> mCompiler;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> mDiagID;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> mDiagOpts;

  std::unique_ptr<LCLJIT> mJIT;
};

Compiler::Compiler(BuildTarget target) {
  mCompiler = std::make_unique<CompilerImpl>(target);
}

void Compiler::addModuleFromSource(const std::string_view source, const std::string_view options) {
  mCompiler->addModuleFromSource(source, options);
}

Compiler::~Compiler() = default;
}
