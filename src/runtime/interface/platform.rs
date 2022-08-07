use super::{ContextKind, DeviceKind};
use crate::api::cl_types::*;
use crate::sync::*;
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::Platform as VkPlatform;

#[cfg(feature = "metal")]
use crate::metal::Platform as MTLPlatform;

#[cfg(test)]
use crate::mock::Platform as MockPlatform;

use crate::cpu::Platform as CPUPlatform;

/// Common interface for Platform objects for all backends.
#[enum_dispatch(PlatformKind)]
pub trait PlatformImpl: ClObjectImpl<cl_platform_id> {
    /// Returns a list of devices, that belong to this platform. Device handles
    /// must be the same, no matter how many times this function is called.
    fn get_devices(&self) -> &[SharedPtr<DeviceKind>];

    // TODO remove this method
    fn add_device(&mut self, device: SharedPtr<DeviceKind>);

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
        devices: &[WeakPtr<DeviceKind>],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> ContextKind;

    // Platform info

    /// Returns supported OpenCL profile: either FULL_PROFILE or EMBEDDED_PROFILE.
    fn get_profile(&self) -> &str;

    /// Returns additional platform version info.
    ///
    /// All LibreCL platforms support OpenCL 3.0. This function allows for
    /// customization of platform version string with additional info.
    /// The returned line will be appended to "OpenCL 3.0 " string.
    fn get_platform_version_info(&self) -> &str;

    /// Returns current platform name. Platform *must* include "LibreCL" prefix
    /// to make sure applications can distinct it from vendors' native platforms.
    fn get_platform_name(&self) -> &str;

    /// Returns a list of platform-wide extension names.
    fn get_extension_names(&self) -> &[&str];

    /// Returns a list of platform-wide extension versions.
    fn get_extension_versions(&self) -> &[cl_version];

    /// Returns host timer resolution for host-device timer synchronization.
    fn get_host_timer_resolution(&self) -> cl_ulong;
}

#[enum_dispatch]
#[repr(C)]
pub enum PlatformKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkPlatform),
    #[cfg(feature = "metal")]
    Metal(MTLPlatform),
    CPU(CPUPlatform),
    #[cfg(test)]
    Mock(MockPlatform),
}

impl ClObjectImpl<cl_platform_id> for PlatformKind {
    fn get_cl_handle(&self) -> cl_platform_id {
        match self {
            #[cfg(feature = "vulkan")]
            PlatformKind::Vulkan(platform) => {
                ClObjectImpl::<cl_platform_id>::get_cl_handle(platform)
            }
            #[cfg(feature = "metal")]
            PlatformKind::Metal(platform) => {
                ClObjectImpl::<cl_platform_id>::get_cl_handle(platform)
            }
            PlatformKind::CPU(platform) => ClObjectImpl::<cl_platform_id>::get_cl_handle(platform),
            #[cfg(test)]
            PlatformKind::Mock(platform) => ClObjectImpl::<cl_platform_id>::get_cl_handle(platform),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_platform_id) {
        match self {
            #[cfg(feature = "vulkan")]
            PlatformKind::Vulkan(platform) => {
                ClObjectImpl::<cl_platform_id>::set_cl_handle(platform, handle)
            }
            #[cfg(feature = "metal")]
            PlatformKind::Metal(platform) => {
                ClObjectImpl::<cl_platform_id>::set_cl_handle(platform, handle)
            }
            PlatformKind::CPU(platform) => {
                ClObjectImpl::<cl_platform_id>::set_cl_handle(platform, handle)
            }
            #[cfg(test)]
            PlatformKind::Mock(platform) => {
                ClObjectImpl::<cl_platform_id>::set_cl_handle(platform, handle)
            }
        }
    }
}
