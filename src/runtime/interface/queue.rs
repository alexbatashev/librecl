use crate::sync::*;
use enum_dispatch::enum_dispatch;

use crate::api::cl_types::{cl_command_queue, ClObjectImpl};
#[cfg(feature = "vulkan")]
use crate::vulkan::InOrderQueue as VkInOrderQueue;

#[cfg(feature = "metal")]
use crate::metal::InOrderQueue as MTLInOrderQueue;

use super::{KernelKind, MemKind};

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
pub trait QueueImpl: ClObjectImpl<cl_command_queue> {
    // TODO return event, todo async
    /// Enqueues asynchronous buffer write command to queue.
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: WeakPtr<MemKind>);
    /// Enqueues asynchronous buffer read command to queue.
    fn enqueue_buffer_read(&self, src: WeakPtr<MemKind>, dst: *mut libc::c_void);
    /// Dispatches kernel for execution in queue.
    fn submit(
        &self,
        kernel: WeakPtr<KernelKind>,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    );
    /// Waits for all submitted tasks to finish.
    fn finish(&self);
}

impl ClObjectImpl<cl_command_queue> for QueueKind {
    fn get_cl_handle(&self) -> cl_command_queue {
        match self {
            #[cfg(feature = "vulkan")]
            QueueKind::VulkanInOrder(queue) => {
                ClObjectImpl::<cl_command_queue>::get_cl_handle(queue)
            }
            #[cfg(feature = "metal")]
            QueueKind::MetalInOrder(queue) => {
                ClObjectImpl::<cl_command_queue>::get_cl_handle(queue)
            }
        }
    }
    fn set_cl_handle(&mut self, handle: cl_command_queue) {
        match self {
            #[cfg(feature = "vulkan")]
            QueueKind::VulkanInOrder(queue) => {
                ClObjectImpl::<cl_command_queue>::set_cl_handle(queue, handle)
            }
            #[cfg(feature = "metal")]
            QueueKind::MetalInOrder(queue) => {
                ClObjectImpl::<cl_command_queue>::set_cl_handle(queue, handle)
            }
        }
    }
}
