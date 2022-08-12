use enum_dispatch::enum_dispatch;

use super::ContextKind;
use crate::api::cl_types::{cl_event, ClObjectImpl};
use crate::sync::WeakPtr;

#[cfg(feature = "vulkan")]
use crate::vulkan::Event as VkEvent;
#[cfg(feature = "vulkan")]
use crate::vulkan::HostToGPUEvent as VkHostEvent;

#[cfg(feature = "metal")]
use crate::metal::Event as MTLEvent;

#[enum_dispatch]
#[repr(C)]
pub enum EventKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkEvent),
    #[cfg(feature = "vulkan")]
    VulkanHost(VkHostEvent),
    #[cfg(feature = "metal")]
    MetalEvent(MTLEvent),
}

/// Common interface for Event objects for all backends.
#[enum_dispatch(EventKind)]
pub trait EventImpl: ClObjectImpl<cl_event> {
    fn get_context(&self) -> WeakPtr<ContextKind>;
}

impl ClObjectImpl<cl_event> for EventKind {
    fn get_cl_handle(&self) -> cl_event {
        match self {
            #[cfg(feature = "vulkan")]
            EventKind::Vulkan(event) => ClObjectImpl::<cl_event>::get_cl_handle(event),
            #[cfg(feature = "vulkan")]
            EventKind::VulkanHost(event) => ClObjectImpl::<cl_event>::get_cl_handle(event),
            #[cfg(feature = "metal")]
            MemKind::MetalSDBuffer(event) => ClObjectImpl::<cl_event>::get_cl_handle(event),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_event) {
        match self {
            #[cfg(feature = "vulkan")]
            EventKind::Vulkan(event) => ClObjectImpl::<cl_event>::set_cl_handle(event, handle),
            #[cfg(feature = "vulkan")]
            EventKind::VulkanHost(event) => ClObjectImpl::<cl_event>::set_cl_handle(event, handle),
            #[cfg(feature = "metal")]
            EventKind::Metal(event) => ClObjectImpl::<cl_mem>::set_cl_handle(event, handle),
        }
    }
}
