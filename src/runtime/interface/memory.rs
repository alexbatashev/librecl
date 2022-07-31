use enum_dispatch::enum_dispatch;

use crate::api::cl_types::{cl_mem, ClObjectImpl};
#[cfg(feature = "vulkan")]
use crate::vulkan::SingleDeviceBuffer as VkSDBuffer;

#[cfg(feature = "metal")]
use crate::metal::SingleDeviceBuffer as MTLSDBuffer;

#[enum_dispatch]
#[repr(C)]
pub enum MemKind {
    #[cfg(feature = "vulkan")]
    VulkanSDBuffer(VkSDBuffer),
    #[cfg(feature = "metal")]
    MetalSDBuffer(MTLSDBuffer),
}

/// Common interface for all Memory objects (buffers, images, etc) for all backends.
#[enum_dispatch(MemKind)]
pub trait MemImpl: ClObjectImpl<cl_mem> {}

impl ClObjectImpl<cl_mem> for MemKind {
    fn get_cl_handle(&self) -> cl_mem {
        match self {
            #[cfg(feature = "vulkan")]
            MemKind::VulkanSDBuffer(mem) => ClObjectImpl::<cl_mem>::get_cl_handle(mem),
            #[cfg(feature = "metal")]
            MemKind::MetalSDBuffer(mem) => ClObjectImpl::<cl_mem>::get_cl_handle(mem),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_mem) {
        match self {
            #[cfg(feature = "vulkan")]
            MemKind::VulkanSDBuffer(mem) => ClObjectImpl::<cl_mem>::set_cl_handle(mem, handle),
            #[cfg(feature = "metal")]
            MemKind::MetalSDBuffer(mem) => ClObjectImpl::<cl_mem>::set_cl_handle(mem, handle),
        }
    }
}
