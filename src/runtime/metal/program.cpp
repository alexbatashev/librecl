#include "program.hpp"
#include "frontend.hpp"

#include <optional>
#include <variant>

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

_cl_program::_cl_program(cl_context ctx, std::string_view program)
    : mContext(ctx) {
  mProgramSource = SourceProgram{std::string{program}};
}

void _cl_program::build(std::span<const cl_device_id> devices,
                        std::span<std::string_view> options,
                        std::optional<callback_t> callback) {
  const auto build = [this]() {
    lcl::FrontendResult res = std::visit(
        overloaded{[this](SourceProgram prog) {
                     return mContext->getClangFE().process(prog.source, {});
                   },
                   [](auto prog) { return lcl::FrontendResult(""); }},
        mProgramSource);

    if (!res.success()) {
      mContext->notifyError(res.error());
      return;
    }

    std::vector<unsigned char> spv = mContext->getMetalBE().compile(res);
  };

  if (!callback.has_value()) {
    build();
  }
}
