use enum_dispatch::enum_dispatch;
use std::rc::Weak;

#[cfg(feature = "vulkan")]
use crate::vulkan::InOrderQueue as VkInOrderQueue;

#[cfg(feature = "metal")]
use crate::metal::InOrderQueue as MTLInOrderQueue;

use super::{MemKind, KernelKind};

#[enum_dispatch]
#[repr(C)]
pub enum QueueKind {
    #[cfg(feature = "vulkan")]
    VulkanInOrder(VkInOrderQueue),
    #[cfg(feature = "metal")]
    MetalInOrder(MTLInOrderQueue),
}

/// Common iterfaces for Queue objects for all backends.
#[enum_dispatch(QueueKind)]
pub trait QueueImpl {
    // TODO return event, todo async
    /// Enqueues asynchronous buffer write command to queue.
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: Weak<MemKind>);
    /// Enqueues asynchronous buffer read command to queue.
    fn enqueue_buffer_read(&self, src: Weak<MemKind>, dst: *mut libc::c_void);
    /// Dispatches kernel for execution in queue.
    fn submit(
        &self,
        kernel: Weak<KernelKind>,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    );
    /// Waits for all submitted tasks to finish.
    fn finish(&self);
}
