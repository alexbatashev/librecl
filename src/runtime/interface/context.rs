use enum_dispatch::enum_dispatch;
use tokio::runtime::Runtime;
use std::rc::{Rc, Weak};
use super::{DeviceKind, MemKind, ProgramKind};

#[cfg(feature = "vulkan")]
use crate::vulkan::Context as VkContext;

#[cfg(feature = "metal")]
use crate::metal::Context as MTLContext;

/// Common interface for Context objects for all backends.
#[enum_dispatch(ContextKind)]
pub trait ContextImpl {
    /// Notifies of user errors, if the context was created with a user callback.
    fn notify_error(&self, message: String);
    /// Returns true if the provided device is part of this context.
    fn has_device(&self, device: &DeviceKind) -> bool;
    /// Returns associated Tokio threading runtime. The runtime is used to
    /// execute service tasks, like data transfers.
    fn get_threading_runtime(&self) -> &Runtime;
    /// Returns a list of devices, that were used to create this context.
    fn get_associated_devices(&self) -> &[Weak<DeviceKind>];
    /// Creates a new program in input state from a raw source code string
    fn create_program_with_source(&self, source: String) -> ProgramKind;
    /// Creates a new buffer, bound to this context.
    fn create_buffer(&mut self, size: usize, flags: cl_mem_flags) -> MemKind;
}

#[enum_dispatch]
#[repr(C)]
pub enum ContextKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkContext),
    #[cfg(feature = "metal")]
    Metal(MTLContext),
}
