use enum_dispatch::enum_dispatch;

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
pub trait MemImpl {}
