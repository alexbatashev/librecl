use crate::api::error_handling::cl_int;
#[derive(Debug, Clone)]
pub enum ClError {
    Success(ErrorDescription),
    DeviceNotFound(ErrorDescription),
    DeviceNotAvailable(ErrorDescription),
    CompilerNotAvailable(ErrorDescription),
    MemObjectAllocationFailure(ErrorDescription),
    OutOfResources(ErrorDescription),
    OutOfHostMemory(ErrorDescription),
    ProfilingInfoNotAvailable(ErrorDescription),
    MemCopyOverlap(ErrorDescription),
    ImageFormatMismatch(ErrorDescription),
    ImageFormatNotSupported(ErrorDescription),
    BuildProgramFailure(ErrorDescription),
    MapFailure(ErrorDescription),
    MisalignedSubBufferOffset(ErrorDescription),
    ExecStatusErrorForEventsInWaitList(ErrorDescription),
    CompileProgramFailure(ErrorDescription),
    LinkerNotAvailable(ErrorDescription),
    LinkProgramFailure(ErrorDescription),
    DevicePartitionFailed(ErrorDescription),
    KernelArgInfoNotAvailable(ErrorDescription),
    InvalidValue(ErrorDescription),
    InvalidDeviceType(ErrorDescription),
    InvalidPlatform(ErrorDescription),
    InvalidDevice(ErrorDescription),
    InvalidContext(ErrorDescription),
    InvalidQueueProperties(ErrorDescription),
    InvalidCommandQueue(ErrorDescription),
    InvalidHostPtr(ErrorDescription),
    InvalidMemObject(ErrorDescription),
    InvalidImageFormatDescriptor(ErrorDescription),
    InvalidImageSize(ErrorDescription),
    InvalidSampler(ErrorDescription),
    InvalidBinary(ErrorDescription),
    InvalidBuildOptions(ErrorDescription),
    InvalidProgram(ErrorDescription),
    InvalidProgramExecutable(ErrorDescription),
    InvalidKernelName(ErrorDescription),
    InvalidKernelDefinition(ErrorDescription),
    InvalidKernel(ErrorDescription),
    InvalidArgIndex(ErrorDescription),
    InvalidArgValue(ErrorDescription),
    InvalidArgSize(ErrorDescription),
    InvalidKernelArgs(ErrorDescription),
    InvalidWorkDimension(ErrorDescription),
    InvalidWorkGroupSize(ErrorDescription),
    InvalidWorkItemSize(ErrorDescription),
    InvalidGlobalOffset(ErrorDescription),
    InvalidEventWaitList(ErrorDescription),
    InvalidEvent(ErrorDescription),
    InvalidOperation(ErrorDescription),
    InvalidGlObject(ErrorDescription),
    InvalidBufferSize(ErrorDescription),
    InvalidMipLevel(ErrorDescription),
    InvalidGlobalWorkSize(ErrorDescription),
    InvalidProperty(ErrorDescription),
    InvalidImageDescriptor(ErrorDescription),
    InvalidCompilerOptions(ErrorDescription),
    InvalidLinkerOptions(ErrorDescription),
    InvalidDevicePartitionCount(ErrorDescription),
    InvalidPipeSize(ErrorDescription),
    InvalidDeviceQueue(ErrorDescription),
    InvalidSpecId(ErrorDescription),
    MaxSizeRestrictionExceeded(ErrorDescription),
}

impl ClError {
    pub fn error_code(&self) -> cl_int {
        match self {
            ClError::Success(_) => 0 as cl_int,
            ClError::DeviceNotFound(_) => -1 as cl_int,
            ClError::DeviceNotAvailable(_) => -2 as cl_int,
            ClError::CompilerNotAvailable(_) => -3 as cl_int,
            ClError::MemObjectAllocationFailure(_) => -4 as cl_int,
            ClError::OutOfResources(_) => -5 as cl_int,
            ClError::OutOfHostMemory(_) => -6 as cl_int,
            ClError::ProfilingInfoNotAvailable(_) => -7 as cl_int,
            ClError::MemCopyOverlap(_) => -8 as cl_int,
            ClError::ImageFormatMismatch(_) => -9 as cl_int,
            ClError::ImageFormatNotSupported(_) => -10 as cl_int,
            ClError::BuildProgramFailure(_) => -11 as cl_int,
            ClError::MapFailure(_) => -12 as cl_int,
            ClError::MisalignedSubBufferOffset(_) => -13 as cl_int,
            ClError::ExecStatusErrorForEventsInWaitList(_) => -14 as cl_int,
            ClError::CompileProgramFailure(_) => -15 as cl_int,
            ClError::LinkerNotAvailable(_) => -16 as cl_int,
            ClError::LinkProgramFailure(_) => -17 as cl_int,
            ClError::DevicePartitionFailed(_) => -18 as cl_int,
            ClError::KernelArgInfoNotAvailable(_) => -19 as cl_int,
            ClError::InvalidValue(_) => -30 as cl_int,
            ClError::InvalidDeviceType(_) => -31 as cl_int,
            ClError::InvalidPlatform(_) => -32 as cl_int,
            ClError::InvalidDevice(_) => -33 as cl_int,
            ClError::InvalidContext(_) => -34 as cl_int,
            ClError::InvalidQueueProperties(_) => -35 as cl_int,
            ClError::InvalidCommandQueue(_) => -36 as cl_int,
            ClError::InvalidHostPtr(_) => -37 as cl_int,
            ClError::InvalidMemObject(_) => -38 as cl_int,
            ClError::InvalidImageFormatDescriptor(_) => -39 as cl_int,
            ClError::InvalidImageSize(_) => -40 as cl_int,
            ClError::InvalidSampler(_) => -41 as cl_int,
            ClError::InvalidBinary(_) => -42 as cl_int,
            ClError::InvalidBuildOptions(_) => -43 as cl_int,
            ClError::InvalidProgram(_) => -44 as cl_int,
            ClError::InvalidProgramExecutable(_) => -45 as cl_int,
            ClError::InvalidKernelName(_) => -46 as cl_int,
            ClError::InvalidKernelDefinition(_) => -47 as cl_int,
            ClError::InvalidKernel(_) => -48 as cl_int,
            ClError::InvalidArgIndex(_) => -49 as cl_int,
            ClError::InvalidArgValue(_) => -50 as cl_int,
            ClError::InvalidArgSize(_) => -51 as cl_int,
            ClError::InvalidKernelArgs(_) => -52 as cl_int,
            ClError::InvalidWorkDimension(_) => -53 as cl_int,
            ClError::InvalidWorkGroupSize(_) => -54 as cl_int,
            ClError::InvalidWorkItemSize(_) => -55 as cl_int,
            ClError::InvalidGlobalOffset(_) => -56 as cl_int,
            ClError::InvalidEventWaitList(_) => -57 as cl_int,
            ClError::InvalidEvent(_) => -58 as cl_int,
            ClError::InvalidOperation(_) => -59 as cl_int,
            ClError::InvalidGlObject(_) => -60 as cl_int,
            ClError::InvalidBufferSize(_) => -61 as cl_int,
            ClError::InvalidMipLevel(_) => -62 as cl_int,
            ClError::InvalidGlobalWorkSize(_) => -63 as cl_int,
            ClError::InvalidProperty(_) => -64 as cl_int,
            ClError::InvalidImageDescriptor(_) => -65 as cl_int,
            ClError::InvalidCompilerOptions(_) => -66 as cl_int,
            ClError::InvalidLinkerOptions(_) => -67 as cl_int,
            ClError::InvalidDevicePartitionCount(_) => -68 as cl_int,
            ClError::InvalidPipeSize(_) => -69 as cl_int,
            ClError::InvalidDeviceQueue(_) => -70 as cl_int,
            ClError::InvalidSpecId(_) => -71 as cl_int,
            ClError::MaxSizeRestrictionExceeded(_) => -72 as cl_int,
        }
    }


    pub fn get_name(&self) -> &str {
        match self {
            ClError::Success(_) => "CL_SUCCESS",
            ClError::DeviceNotFound(_) => "CL_DEVICE_NOT_FOUND",
            ClError::DeviceNotAvailable(_) => "CL_DEVICE_NOT_AVAILABLE",
            ClError::CompilerNotAvailable(_) => "CL_COMPILER_NOT_AVAILABLE",
            ClError::MemObjectAllocationFailure(_) => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            ClError::OutOfResources(_) => "CL_OUT_OF_RESOURCES",
            ClError::OutOfHostMemory(_) => "CL_OUT_OF_HOST_MEMORY",
            ClError::ProfilingInfoNotAvailable(_) => "CL_PROFILING_INFO_NOT_AVAILABLE",
            ClError::MemCopyOverlap(_) => "CL_MEM_COPY_OVERLAP",
            ClError::ImageFormatMismatch(_) => "CL_IMAGE_FORMAT_MISMATCH",
            ClError::ImageFormatNotSupported(_) => "CL_IMAGE_FORMAT_NOT_SUPPORTED",
            ClError::BuildProgramFailure(_) => "CL_BUILD_PROGRAM_FAILURE",
            ClError::MapFailure(_) => "CL_MAP_FAILURE",
            ClError::MisalignedSubBufferOffset(_) => "CL_MISALIGNED_SUB_BUFFER_OFFSET",
            ClError::ExecStatusErrorForEventsInWaitList(_) => "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            ClError::CompileProgramFailure(_) => "CL_COMPILE_PROGRAM_FAILURE",
            ClError::LinkerNotAvailable(_) => "CL_LINKER_NOT_AVAILABLE",
            ClError::LinkProgramFailure(_) => "CL_LINK_PROGRAM_FAILURE",
            ClError::DevicePartitionFailed(_) => "CL_DEVICE_PARTITION_FAILED",
            ClError::KernelArgInfoNotAvailable(_) => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
            ClError::InvalidValue(_) => "CL_INVALID_VALUE",
            ClError::InvalidDeviceType(_) => "CL_INVALID_DEVICE_TYPE",
            ClError::InvalidPlatform(_) => "CL_INVALID_PLATFORM",
            ClError::InvalidDevice(_) => "CL_INVALID_DEVICE",
            ClError::InvalidContext(_) => "CL_INVALID_CONTEXT",
            ClError::InvalidQueueProperties(_) => "CL_INVALID_QUEUE_PROPERTIES",
            ClError::InvalidCommandQueue(_) => "CL_INVALID_COMMAND_QUEUE",
            ClError::InvalidHostPtr(_) => "CL_INVALID_HOST_PTR",
            ClError::InvalidMemObject(_) => "CL_INVALID_MEM_OBJECT",
            ClError::InvalidImageFormatDescriptor(_) => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
            ClError::InvalidImageSize(_) => "CL_INVALID_IMAGE_SIZE",
            ClError::InvalidSampler(_) => "CL_INVALID_SAMPLER",
            ClError::InvalidBinary(_) => "CL_INVALID_BINARY",
            ClError::InvalidBuildOptions(_) => "CL_INVALID_BUILD_OPTIONS",
            ClError::InvalidProgram(_) => "CL_INVALID_PROGRAM",
            ClError::InvalidProgramExecutable(_) => "CL_INVALID_PROGRAM_EXECUTABLE",
            ClError::InvalidKernelName(_) => "CL_INVALID_KERNEL_NAME",
            ClError::InvalidKernelDefinition(_) => "CL_INVALID_KERNEL_DEFINITION",
            ClError::InvalidKernel(_) => "CL_INVALID_KERNEL",
            ClError::InvalidArgIndex(_) => "CL_INVALID_ARG_INDEX",
            ClError::InvalidArgValue(_) => "CL_INVALID_ARG_VALUE",
            ClError::InvalidArgSize(_) => "CL_INVALID_ARG_SIZE",
            ClError::InvalidKernelArgs(_) => "CL_INVALID_KERNEL_ARGS",
            ClError::InvalidWorkDimension(_) => "CL_INVALID_WORK_DIMENSION",
            ClError::InvalidWorkGroupSize(_) => "CL_INVALID_WORK_GROUP_SIZE",
            ClError::InvalidWorkItemSize(_) => "CL_INVALID_WORK_ITEM_SIZE",
            ClError::InvalidGlobalOffset(_) => "CL_INVALID_GLOBAL_OFFSET",
            ClError::InvalidEventWaitList(_) => "CL_INVALID_EVENT_WAIT_LIST",
            ClError::InvalidEvent(_) => "CL_INVALID_EVENT",
            ClError::InvalidOperation(_) => "CL_INVALID_OPERATION",
            ClError::InvalidGlObject(_) => "CL_INVALID_GL_OBJECT",
            ClError::InvalidBufferSize(_) => "CL_INVALID_BUFFER_SIZE",
            ClError::InvalidMipLevel(_) => "CL_INVALID_MIP_LEVEL",
            ClError::InvalidGlobalWorkSize(_) => "CL_INVALID_GLOBAL_WORK_SIZE",
            ClError::InvalidProperty(_) => "CL_INVALID_PROPERTY",
            ClError::InvalidImageDescriptor(_) => "CL_INVALID_IMAGE_DESCRIPTOR",
            ClError::InvalidCompilerOptions(_) => "CL_INVALID_COMPILER_OPTIONS",
            ClError::InvalidLinkerOptions(_) => "CL_INVALID_LINKER_OPTIONS",
            ClError::InvalidDevicePartitionCount(_) => "CL_INVALID_DEVICE_PARTITION_COUNT",
            ClError::InvalidPipeSize(_) => "CL_INVALID_PIPE_SIZE",
            ClError::InvalidDeviceQueue(_) => "CL_INVALID_DEVICE_QUEUE",
            ClError::InvalidSpecId(_) => "CL_INVALID_SPEC_ID",
            ClError::MaxSizeRestrictionExceeded(_) => "CL_MAX_SIZE_RESTRICTION_EXCEEDED",
        }
    }
}

impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error_descr = match self {
            ClError::Success(desc) => desc,
            ClError::DeviceNotFound(desc) => desc,
            ClError::DeviceNotAvailable(desc) => desc,
            ClError::CompilerNotAvailable(desc) => desc,
            ClError::MemObjectAllocationFailure(desc) => desc,
            ClError::OutOfResources(desc) => desc,
            ClError::OutOfHostMemory(desc) => desc,
            ClError::ProfilingInfoNotAvailable(desc) => desc,
            ClError::MemCopyOverlap(desc) => desc,
            ClError::ImageFormatMismatch(desc) => desc,
            ClError::ImageFormatNotSupported(desc) => desc,
            ClError::BuildProgramFailure(desc) => desc,
            ClError::MapFailure(desc) => desc,
            ClError::MisalignedSubBufferOffset(desc) => desc,
            ClError::ExecStatusErrorForEventsInWaitList(desc) => desc,
            ClError::CompileProgramFailure(desc) => desc,
            ClError::LinkerNotAvailable(desc) => desc,
            ClError::LinkProgramFailure(desc) => desc,
            ClError::DevicePartitionFailed(desc) => desc,
            ClError::KernelArgInfoNotAvailable(desc) => desc,
            ClError::InvalidValue(desc) => desc,
            ClError::InvalidDeviceType(desc) => desc,
            ClError::InvalidPlatform(desc) => desc,
            ClError::InvalidDevice(desc) => desc,
            ClError::InvalidContext(desc) => desc,
            ClError::InvalidQueueProperties(desc) => desc,
            ClError::InvalidCommandQueue(desc) => desc,
            ClError::InvalidHostPtr(desc) => desc,
            ClError::InvalidMemObject(desc) => desc,
            ClError::InvalidImageFormatDescriptor(desc) => desc,
            ClError::InvalidImageSize(desc) => desc,
            ClError::InvalidSampler(desc) => desc,
            ClError::InvalidBinary(desc) => desc,
            ClError::InvalidBuildOptions(desc) => desc,
            ClError::InvalidProgram(desc) => desc,
            ClError::InvalidProgramExecutable(desc) => desc,
            ClError::InvalidKernelName(desc) => desc,
            ClError::InvalidKernelDefinition(desc) => desc,
            ClError::InvalidKernel(desc) => desc,
            ClError::InvalidArgIndex(desc) => desc,
            ClError::InvalidArgValue(desc) => desc,
            ClError::InvalidArgSize(desc) => desc,
            ClError::InvalidKernelArgs(desc) => desc,
            ClError::InvalidWorkDimension(desc) => desc,
            ClError::InvalidWorkGroupSize(desc) => desc,
            ClError::InvalidWorkItemSize(desc) => desc,
            ClError::InvalidGlobalOffset(desc) => desc,
            ClError::InvalidEventWaitList(desc) => desc,
            ClError::InvalidEvent(desc) => desc,
            ClError::InvalidOperation(desc) => desc,
            ClError::InvalidGlObject(desc) => desc,
            ClError::InvalidBufferSize(desc) => desc,
            ClError::InvalidMipLevel(desc) => desc,
            ClError::InvalidGlobalWorkSize(desc) => desc,
            ClError::InvalidProperty(desc) => desc,
            ClError::InvalidImageDescriptor(desc) => desc,
            ClError::InvalidCompilerOptions(desc) => desc,
            ClError::InvalidLinkerOptions(desc) => desc,
            ClError::InvalidDevicePartitionCount(desc) => desc,
            ClError::InvalidPipeSize(desc) => desc,
            ClError::InvalidDeviceQueue(desc) => desc,
            ClError::InvalidSpecId(desc) => desc,
            ClError::MaxSizeRestrictionExceeded(desc) => desc,
        };
        write!(f, "{} ({}) - {} from:\n{:?}", self.error_code(), self.get_name(), error_descr.message, error_descr.backtrace)
    }
}
#[cfg(test)]
pub mod error_codes {
use crate::api::error_handling::cl_int;
    pub const CL_SUCCESS: cl_int = 0;
    pub const CL_DEVICE_NOT_FOUND: cl_int = -1;
    pub const CL_DEVICE_NOT_AVAILABLE: cl_int = -2;
    pub const CL_COMPILER_NOT_AVAILABLE: cl_int = -3;
    pub const CL_MEM_OBJECT_ALLOCATION_FAILURE: cl_int = -4;
    pub const CL_OUT_OF_RESOURCES: cl_int = -5;
    pub const CL_OUT_OF_HOST_MEMORY: cl_int = -6;
    pub const CL_PROFILING_INFO_NOT_AVAILABLE: cl_int = -7;
    pub const CL_MEM_COPY_OVERLAP: cl_int = -8;
    pub const CL_IMAGE_FORMAT_MISMATCH: cl_int = -9;
    pub const CL_IMAGE_FORMAT_NOT_SUPPORTED: cl_int = -10;
    pub const CL_BUILD_PROGRAM_FAILURE: cl_int = -11;
    pub const CL_MAP_FAILURE: cl_int = -12;
    pub const CL_MISALIGNED_SUB_BUFFER_OFFSET: cl_int = -13;
    pub const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: cl_int = -14;
    pub const CL_COMPILE_PROGRAM_FAILURE: cl_int = -15;
    pub const CL_LINKER_NOT_AVAILABLE: cl_int = -16;
    pub const CL_LINK_PROGRAM_FAILURE: cl_int = -17;
    pub const CL_DEVICE_PARTITION_FAILED: cl_int = -18;
    pub const CL_KERNEL_ARG_INFO_NOT_AVAILABLE: cl_int = -19;
    pub const CL_INVALID_VALUE: cl_int = -30;
    pub const CL_INVALID_DEVICE_TYPE: cl_int = -31;
    pub const CL_INVALID_PLATFORM: cl_int = -32;
    pub const CL_INVALID_DEVICE: cl_int = -33;
    pub const CL_INVALID_CONTEXT: cl_int = -34;
    pub const CL_INVALID_QUEUE_PROPERTIES: cl_int = -35;
    pub const CL_INVALID_COMMAND_QUEUE: cl_int = -36;
    pub const CL_INVALID_HOST_PTR: cl_int = -37;
    pub const CL_INVALID_MEM_OBJECT: cl_int = -38;
    pub const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: cl_int = -39;
    pub const CL_INVALID_IMAGE_SIZE: cl_int = -40;
    pub const CL_INVALID_SAMPLER: cl_int = -41;
    pub const CL_INVALID_BINARY: cl_int = -42;
    pub const CL_INVALID_BUILD_OPTIONS: cl_int = -43;
    pub const CL_INVALID_PROGRAM: cl_int = -44;
    pub const CL_INVALID_PROGRAM_EXECUTABLE: cl_int = -45;
    pub const CL_INVALID_KERNEL_NAME: cl_int = -46;
    pub const CL_INVALID_KERNEL_DEFINITION: cl_int = -47;
    pub const CL_INVALID_KERNEL: cl_int = -48;
    pub const CL_INVALID_ARG_INDEX: cl_int = -49;
    pub const CL_INVALID_ARG_VALUE: cl_int = -50;
    pub const CL_INVALID_ARG_SIZE: cl_int = -51;
    pub const CL_INVALID_KERNEL_ARGS: cl_int = -52;
    pub const CL_INVALID_WORK_DIMENSION: cl_int = -53;
    pub const CL_INVALID_WORK_GROUP_SIZE: cl_int = -54;
    pub const CL_INVALID_WORK_ITEM_SIZE: cl_int = -55;
    pub const CL_INVALID_GLOBAL_OFFSET: cl_int = -56;
    pub const CL_INVALID_EVENT_WAIT_LIST: cl_int = -57;
    pub const CL_INVALID_EVENT: cl_int = -58;
    pub const CL_INVALID_OPERATION: cl_int = -59;
    pub const CL_INVALID_GL_OBJECT: cl_int = -60;
    pub const CL_INVALID_BUFFER_SIZE: cl_int = -61;
    pub const CL_INVALID_MIP_LEVEL: cl_int = -62;
    pub const CL_INVALID_GLOBAL_WORK_SIZE: cl_int = -63;
    pub const CL_INVALID_PROPERTY: cl_int = -64;
    pub const CL_INVALID_IMAGE_DESCRIPTOR: cl_int = -65;
    pub const CL_INVALID_COMPILER_OPTIONS: cl_int = -66;
    pub const CL_INVALID_LINKER_OPTIONS: cl_int = -67;
    pub const CL_INVALID_DEVICE_PARTITION_COUNT: cl_int = -68;
    pub const CL_INVALID_PIPE_SIZE: cl_int = -69;
    pub const CL_INVALID_DEVICE_QUEUE: cl_int = -70;
    pub const CL_INVALID_SPEC_ID: cl_int = -71;
    pub const CL_MAX_SIZE_RESTRICTION_EXCEEDED: cl_int = -72;
}

