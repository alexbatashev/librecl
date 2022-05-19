#include <iostream>
#include <string_view>

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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "compiler.hpp"
#include "transformations.hpp"

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
  }

  void addModuleFromSource(const std::string_view source,
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

    llvm::StringRef irModule(static_cast<const char *>(irBuffer.data()),
                             irBuffer.size());

    std::unique_ptr<llvm::MemoryBuffer> MB =
        llvm::MemoryBuffer::getMemBuffer(irModule, "sample.bc", false);

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
        IRCleanup cleanup;
        AIRLegalize legalize{M->getContext()};

        M->setTargetTriple("air64-apple-macosx12.0.0");
        M->setDataLayout(
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:"
            "32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-"
            "v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-"
            "v1024:1024:1024-n8:16:32");
        // cleanup.apply(*M);
        M = legalize.apply(std::move(M));
        M->print(llvm::outs(), nullptr);
      }
    } else
      std::cout << errString << "\n";
  }

private:
  std::unique_ptr<clang::CompilerInstance> mCompiler;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> mDiagID;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> mDiagOpts;
};

Compiler::Compiler(BuildTarget target) {
  mCompiler = std::make_unique<CompilerImpl>(target);
}

void Compiler::addModuleFromSource(const std::string_view source,
                                   std::span<std::string_view> options) {
  llvm::SmallVector<llvm::StringRef, 15> opts;
  for (std::string_view opt : options) {
    opts.push_back(llvm::StringRef{opt});
  }
  mCompiler->addModuleFromSource(source, opts);
}

Compiler::~Compiler() = default;
} // namespace lcl
