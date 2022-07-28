use enum_dispatch::enum_dispatch;
use super::DeviceKind;
use std::rc::{Rc, Weak};

#[cfg(feature = "vulkan")]
use crate::vulkan::Platform as VkPlatform;

#[cfg(feature = "metal")]
use crate::metal::Platform as MTLPlatform;

#[cfg(test)]
use crate::mock::Platform as MockPlatform;

/// Common interface for Platform objects for all backends.
#[enum_dispatch(PlatformKind)]
pub trait PlatformImpl {
    /// Returns current platform name. Platform *must* include "LibreCL" prefix
    /// to make sure applications can distinct it from vendors' native platforms.
    fn get_platform_name(&self) -> &str;

    /// Returns a list of devices, that belong to this platform. Device handles
    /// must be the same, no matter how many times this function is called.
    fn get_devices(&self) -> &[Rc<DeviceKind>];

    // TODO remove this method
    fn add_device(&mut self, device: Rc<DeviceKind>);

    /// Creates a new context.
    ///
    /// # Arguments
    /// 
    /// * `devices` - list of devices to be included in context. Caller must
    ///    ensure, that all devices belong to this platform. Implementation
    ///    may add more constraints to whether devices can be put in the same context.
    /// * `callback` - user function, that is called any time an error happens
    ///    on the context. This allows for a more friendly error feedback, than
    ///    just the error codes.
    /// * `user_data` - a pointer to user data, that will be passed to the callback.
    ///    It is user responsibility to manage lifetime and access to this data.
    fn create_context(
        &self,
        devices: &[Weak<DeviceKind>],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> cl_context;
}

#[enum_dispatch]
#[repr(C)]
pub enum PlatformKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkPlatform),
    #[cfg(feature = "metal")]
    Metal(MTLPlatform),
    #[cfg(test)]
    Mock(MockPlatform),
}
