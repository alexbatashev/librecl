#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>

namespace lcl {
class LCLJIT {
public:
  LCLJIT(std::unique_ptr<llvm::orc::ExecutionSession> ES,
                  llvm::orc::JITTargetMachineBuilder JTMB, llvm::DataLayout DL)
      : mES(std::move(ES)), mDL(std::move(DL)), mMangle(*mES, mDL),
        mObjectLayer(*mES,
                    []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
        mCompileLayer(*mES, mObjectLayer,
                     std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
        MainJD(mES->createBareJITDylib("<main>")) {
    MainJD.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            mDL.getGlobalPrefix())));
  }

  ~LCLJIT() {
    if (auto Err = mES->endSession())
      mES->reportError(std::move(Err));
  }

  static llvm::Expected<std::unique_ptr<LCLJIT>> create() {
    auto EPC = llvm::orc::SelfExecutorProcessControl::Create();
    if (!EPC)
      return EPC.takeError();

    auto ES = std::make_unique<llvm::orc::ExecutionSession>(std::move(*EPC));

    llvm::orc::JITTargetMachineBuilder JTMB(
        ES->getExecutorProcessControl().getTargetTriple());

    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<LCLJIT>(std::move(ES), std::move(JTMB),
                                             std::move(*DL));
  }

  const llvm::DataLayout &getDataLayout() const { return mDL; }

  llvm::orc::JITDylib &getMainJITDylib() { return MainJD; }

  llvm::Error addModule(llvm::orc::ThreadSafeModule TSM, llvm::orc::ResourceTrackerSP RT = nullptr) {
    if (!RT)
      RT = MainJD.getDefaultResourceTracker();
    return mCompileLayer.add(RT, std::move(TSM));
  }

  llvm::Expected<llvm::JITEvaluatedSymbol> lookup(llvm::StringRef Name) {
    return mES->lookup({&MainJD}, mMangle(Name.str()));
  }
private:
  std::unique_ptr<llvm::orc::ExecutionSession> mES;
  llvm::orc::RTDyldObjectLinkingLayer mObjectLayer;
  llvm::orc::IRCompileLayer mCompileLayer;

  llvm::DataLayout mDL;
  llvm::orc::MangleAndInterner mMangle;
  llvm::orc::ThreadSafeContext mCtx;

  llvm::orc::JITDylib &MainJD;
};
}
