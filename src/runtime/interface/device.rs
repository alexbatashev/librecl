use super::{ContextKind, PlatformKind, QueueKind};
use crate::{
    api::cl_types::*,
    sync::{SharedPtr, WeakPtr},
};
use enum_dispatch::enum_dispatch;

#[cfg(feature = "metal")]
use crate::metal::Device as MTLDevice;
#[cfg(feature = "vulkan")]
use crate::vulkan::Device as VkDevice;

use crate::cpu::Device as CPUDevice;

#[enum_dispatch]
#[repr(C)]
pub enum DeviceKind {
    #[cfg(feature = "vulkan")]
    Vulkan(VkDevice),
    #[cfg(feature = "metal")]
    Metal(MTLDevice),
    CPU(CPUDevice),
}

// TODO do we still need this?!
unsafe impl Sync for DeviceKind {}

/// Common interface for Device objects for all backends.
#[enum_dispatch(DeviceKind)]
pub trait DeviceImpl: ClObjectImpl<cl_device_id> {
    /// Returns OpenCL device type.
    fn get_device_type(&self) -> cl_device_type;

    /// Returns reference to the platform this device belongs to.
    fn get_platform(&self) -> WeakPtr<PlatformKind>;

    /// Creates a new command queue for this device.
    fn create_queue(
        &self,
        context: SharedPtr<ContextKind>,
        device: SharedPtr<DeviceKind>,
    ) -> QueueKind;

    // GetDeviceInfo stubs

    /// Returns device name. The device name should not contain "LibreCL" prefix
    /// and sould match vendors' device name.
    fn get_device_name(&self) -> String;

    /// Returns device vendor. This should not indicate, that the device is
    /// exposed through LibreCL library.
    fn get_vendor_name(&self) -> String;

    /// Returns numeric device vendor ID. If the vendor has a PCI vendor ID,
    /// the low 16 bits must contain that PCI vendor ID, and the remaining bits
    /// must be set to zero. Otherwise, the value returned must be a valid
    /// Khronos vendor ID represented by type cl_khronos_vendor_id. Khronos
    /// vendor IDs are allocated starting at 0x10000, to distinguish them from
    /// the PCI vendor ID namespace.
    fn get_vendor_id(&self) -> cl_uint;

    /// Returns maximum number of parallel compute units on OpenCL device.
    /// Must be at least 1.
    fn get_max_compute_units(&self) -> cl_uint;

    /// Returns maximum number of ND Range dimensions. Must be 3 for non-custom
    /// devices.
    fn get_max_work_item_dimensions(&self) -> cl_uint;

    /// Returns maximum size of ND Range for each dimension.
    fn get_max_work_item_sizes(&self) -> [cl_size_t; 3];

    /// Some devices (like eGPUs) can be physically unplugged while application
    /// is still running. This function must return true if the device is still
    /// available and can be used.
    fn is_available(&self) -> bool;

    /// Returns true if compiler and linker are available.
    /// Compiler is distributed as a separate library, and may be missing in some
    /// cases, e.g. when distributing a mobile application. In that case, it
    /// makes sense to pre-compile everything with `lcloc` tool to save both
    /// space and compile times.
    fn is_compiler_available(&self) -> bool;

    /// Returns native driver version in vendor-specific format.
    fn get_native_driver_version(&self) -> String;

    /// Returns FULL_PROFILE or EMBEDDED_PROFILE. When compiler is not available,
    /// must return EMBEDDED_PROFILE.
    fn get_device_profile(&self) -> String;

    /// Returns device-specific version info.
    fn get_device_version_info(&self) -> String;

    /// Returns a slice of supported device extensions.
    fn get_extension_names(&self) -> &[&str];

    /// Returns extensions versions in the same order as extension names.
    fn get_extension_versions(&self) -> &[cl_version];
}

impl ClObjectImpl<cl_device_id> for DeviceKind {
    fn get_cl_handle(&self) -> cl_device_id {
        match self {
            #[cfg(feature = "vulkan")]
            DeviceKind::Vulkan(device) => ClObjectImpl::<cl_device_id>::get_cl_handle(device),
            #[cfg(feature = "metal")]
            DeviceKind::Metal(device) => ClObjectImpl::<cl_device_id>::get_cl_handle(device),
            DeviceKind::CPU(device) => ClObjectImpl::<cl_device_id>::get_cl_handle(device),
        }
    }
    fn set_cl_handle(&mut self, handle: cl_device_id) {
        match self {
            #[cfg(feature = "vulkan")]
            DeviceKind::Vulkan(device) => {
                ClObjectImpl::<cl_device_id>::set_cl_handle(device, handle)
            }
            #[cfg(feature = "metal")]
            DeviceKind::Metal(device) => {
                ClObjectImpl::<cl_device_id>::set_cl_handle(device, handle)
            }
            DeviceKind::CPU(device) => ClObjectImpl::<cl_device_id>::set_cl_handle(device, handle),
        }
    }
}
