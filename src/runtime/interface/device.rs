use enum_dispatch::enum_dispatch;
use std::rc::{Rc, Weak};
use super::{PlatformKind, ContextKind, QueueKind};

#[cfg(feature = "metal")]
use crate::metal::Device as MTLDevice;
#[cfg(feature = "vulkan")]
use crate::vulkan::Device as VkDevice;

#[enum_dispatch]
#[repr(C)]
pub enum DeviceKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkDevice),
    #[cfg(feature = "metal")]
    Metal(MTLDevice),
}

// TODO do we still need this?!
unsafe impl Sync for DeviceKind {}

/// Common interface for Device objects for all backends.
#[enum_dispatch(DeviceKind)]
pub trait DeviceImpl: Sync {
    /// Returns OpenCL device type.
    fn get_device_type(&self) -> cl_device_type;
    /// Some devices (like eGPUs) can be physically unplugged while application
    /// is still running. This function must return true if the device is still
    /// available and can be used.
    fn is_available(&self) -> bool;
    /// Returns reference to the platform this device belongs to.
    fn get_platform(&self) -> Weak<PlatformKind>;
    /// Creates a new command queue for this device.
    fn create_queue(&self, context: Rc<ContextKind>, device: Rc<DeviceKind>) -> QueueKind;

    // GetDeviceInfo stubs

    /// Returns device name. The device name should not contain "LibreCL" prefix
    /// and sould match vendors' device name.
    fn get_device_name(&self) -> String;
}

