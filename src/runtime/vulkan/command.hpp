#pragma once

#include <CL/cl.h>
#include <vulkan/vulkan.hpp>

#include <span>

class Command {
public:
  virtual ~Command() = default;

  enum class EnqueueType { Blocking, NonBlocking };

  virtual cl_event recordCommand(cl_command_queue queue,
                                 vk::CommandBuffer commandBuffer) = 0;

protected:
  enum class CommandType { MemWriteBuffer, MemReadBuffer };

  Command(CommandType type, EnqueueType enqType)
      : mCommandType(type), mEnqueueType(enqType){};

  CommandType mCommandType;
  EnqueueType mEnqueueType;
};

class MemWriteBufferCommand : public Command {
public:
  MemWriteBufferCommand(cl_mem buffer, EnqueueType type, size_t offset,
                        size_t size, const void *ptr,
                        std::span<cl_event> waitList);

  ~MemWriteBufferCommand() override = default;

  cl_event recordCommand(cl_command_queue queue,
                         vk::CommandBuffer commandBuffer) override;

private:
  cl_mem mDst;
  const void *mSrc;
  size_t mOffset;
  size_t mSize;
  std::vector<cl_event> mWaitList;
};
