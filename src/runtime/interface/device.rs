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

#[derive(Debug, Clone, Copy)]
pub struct VectorCaps {
    pub vector_width_char: cl_uint,
    pub vector_width_short: cl_uint,
    pub vector_width_int: cl_uint,
    pub vector_width_long: cl_uint,
    pub vector_width_float: cl_uint,
    pub vector_width_double: cl_uint,
    pub vector_width_half: cl_uint,
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceLimits {
    pub max_compute_units: cl_uint,
    pub max_work_item_dimensions: cl_uint,
    pub max_work_item_sizes: [cl_size_t; 3],
    pub max_work_group_size: cl_size_t,
    pub preferred_vector_caps: VectorCaps,
    pub native_vector_caps: VectorCaps,
    pub max_mem_alloc_size: cl_ulong,
    pub preferred_work_group_size_multiple: cl_size_t,
    // TODO max parameter size
    // TODO other numeric limits
}

pub trait DeviceLimitsInterface {
    /// Returns maximum number of parallel compute units on OpenCL device.
    /// Must be at least 1.
    fn max_compute_units(&self) -> cl_uint;
    /// Returns maximum number of ND Range dimensions. Must be 3 for non-custom
    /// devices.
    fn max_work_item_dimensions(&self) -> cl_uint;
    /// Returns maximum size of ND Range for each dimension.
    fn max_work_item_sizes(&self) -> [cl_size_t; 3];
    fn max_work_group_size(&self) -> cl_size_t;
    fn preferred_vector_width_char(&self) -> cl_uint;
    fn preferred_vector_width_short(&self) -> cl_uint;
    fn preferred_vector_width_int(&self) -> cl_uint;
    fn preferred_vector_width_long(&self) -> cl_uint;
    fn preferred_vector_width_float(&self) -> cl_uint;
    fn preferred_vector_width_double(&self) -> cl_uint;
    fn preferred_vector_width_half(&self) -> cl_uint;
    fn native_vector_width_char(&self) -> cl_uint;
    fn native_vector_width_short(&self) -> cl_uint;
    fn native_vector_width_int(&self) -> cl_uint;
    fn native_vector_width_long(&self) -> cl_uint;
    fn native_vector_width_float(&self) -> cl_uint;
    fn native_vector_width_double(&self) -> cl_uint;
    fn native_vector_width_half(&self) -> cl_uint;
    fn max_mem_alloc_size(&self) -> cl_ulong;
    fn preferred_work_group_size_multiple(&self) -> cl_size_t;
}

#[enum_dispatch]
#[repr(C)]
#[derive(ocl_type_wrapper::DeviceLimitsInterface)]
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
pub trait DeviceImpl: ClObjectImpl<cl_device_id> + DeviceLimitsInterface {
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
