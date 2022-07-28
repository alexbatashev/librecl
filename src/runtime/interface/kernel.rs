use enum_dispatch::enum_dispatch;
use librecl_compiler::KernelArgInfo;
use std::rc::Weak;
use super::MemKind;

#[cfg(feature = "vulkan")]
use crate::vulkan::Kernel as VkKernel;

#[cfg(feature = "metal")]
use crate::metal::Kernel as MTLKernel;

#[enum_dispatch]
#[repr(C)]
pub enum KernelKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkKernel),
    #[cfg(feature = "metal")]
    Metal(MTLKernel),
}

/// Common interface for Kernel objects for all backends.
#[enum_dispatch(ClKernel)]
pub trait KernelImpl {
    /// Sets Plain Old Data argument for this kernel object.
    ///
    /// # Arguments
    ///
    /// * `index` is index of argument to be set in range [0; n).
    /// * `bytes` is a list of raw data bytes to be set
    fn set_data_arg(&mut self, index: usize, bytes: &[u8]);
    /// Sets buffer argument for this object.
    ///
    /// # Arguments
    ///
    /// * `index` is index of argument to be set in range [0; n).
    /// * `buffer` is a LibreCL buffer, bound to the same context as this kernel.
    fn set_buffer_arg(&mut self, index: usize, buffer: Weak<MemKind>);
    /// Returns descriptors of all the kernel arguments.
    fn get_arg_info(&self) -> &[KernelArgInfo];
}
