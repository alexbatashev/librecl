#pragma once

#include <CL/cl.h>
#include <vulkan/vulkan.hpp>

#include <array>
#include <span>

class Command {
public:
  virtual ~Command() = default;

  enum class EnqueueType { Blocking, NonBlocking };

  virtual cl_event recordCommand(cl_command_queue queue,
                                 vk::CommandBuffer commandBuffer) = 0;

protected:
  Command(EnqueueType enqType) : mEnqueueType(enqType){};

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

class MemReadBufferCommand : public Command {
public:
  MemReadBufferCommand(cl_mem buffer, EnqueueType type, size_t offset,
                       size_t size, void *ptr, std::span<cl_event> waitList);

  ~MemReadBufferCommand() override = default;

  cl_event recordCommand(cl_command_queue queue,
                         vk::CommandBuffer commandBuffer) override;

private:
  cl_mem mSrc;
  void *mDst;
  size_t mOffset;
  size_t mSize;
  std::vector<cl_event> mWaitList;
};

class ExecKernelCommand : public Command {
public:
  struct NDRange {
    std::array<size_t, 3> globalOffset;
    std::array<size_t, 3> globalSize;
    std::array<size_t, 3> localSize;
  };

  ExecKernelCommand(cl_kernel kernel, NDRange range,
                    std::span<const cl_event> waitList)
      : mKernel(kernel), mRange(range),
        mWaitList(waitList.begin(), waitList.end()),
        Command(EnqueueType::NonBlocking){};

  cl_event recordCommand(cl_command_queue queue,
                         vk::CommandBuffer commandBuffer) override;

private:
  cl_kernel mKernel;
  NDRange mRange;
  std::span<const cl_event> mWaitList;
};
