use super::MemKind;
use crate::api::cl_types::cl_kernel;
use enum_dispatch::enum_dispatch;
use librecl_compiler::KernelArgInfo;

use crate::api::cl_types::ClObjectImpl;
use crate::sync::WeakPtr;
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
#[enum_dispatch(KernelKind)]
pub trait KernelImpl: ClObjectImpl<cl_kernel> {
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
    fn set_buffer_arg(&mut self, index: usize, buffer: WeakPtr<MemKind>);
    /// Returns descriptors of all the kernel arguments.
    fn get_arg_info(&self) -> &[KernelArgInfo];
}

impl ClObjectImpl<cl_kernel> for KernelKind {
    fn get_cl_handle(&self) -> cl_kernel {
        match self {
            #[cfg(feature = "vulkan")]
            KernelKind::Vulkan(kernel) => ClObjectImpl::<cl_kernel>::get_cl_handle(kernel),
            #[cfg(feature = "metal")]
            KernelKind::Metal(kernel) => ClObjectImpl::<cl_kernel>::get_cl_handle(kernel),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_kernel) {
        match self {
            #[cfg(feature = "vulkan")]
            KernelKind::Vulkan(kernel) => ClObjectImpl::<cl_kernel>::set_cl_handle(kernel, handle),
            #[cfg(feature = "metal")]
            KernelKind::Metal(kernel) => ClObjectImpl::<cl_kernel>::set_cl_handle(kernel, handle),
        }
    }
}
