use crate::api::cl_types::ClObjectImpl;
use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceImpl;
use crate::interface::DeviceKind;
use crate::interface::PlatformKind;
use crate::interface::QueueKind;
use crate::interface::{DeviceLimits, VectorCaps};
use crate::sync::UnsafeHandle;
use crate::sync::{self, *};
use lcl_derive::ClObjImpl;
use lcl_derive::DeviceLimitsInterface;
use pytorch_cpuinfo::Package;

#[derive(ClObjImpl, DeviceLimitsInterface)]
pub struct Device {
    device_type: cl_device_type,
    name: String,
    platform: WeakPtr<PlatformKind>,
    device_limits: DeviceLimits,
    #[cl_handle]
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(platform: WeakPtr<PlatformKind>, package: &Package) -> SharedPtr<DeviceKind> {
        // TODO figure out real limits
        let vec_limits = VectorCaps {
            vector_width_char: 1,
            vector_width_short: 1,
            vector_width_int: 1,
            vector_width_long: 1,
            vector_width_float: 1,
            vector_width_double: 1,
            vector_width_half: 1,
        };

        let device_limits = DeviceLimits {
            max_compute_units: 1,
            max_work_item_dimensions: 3,
            max_work_item_sizes: [0, 0, 0],
            max_work_group_size: 0,
            preferred_vector_caps: vec_limits.clone(),
            native_vector_caps: vec_limits,
            max_mem_alloc_size: 0,
            preferred_work_group_size_multiple: 0,
        };

        let device = Device {
            device_type: cl_device_type::CPU,
            name: package.name.clone(),
            platform,
            device_limits,
            handle: UnsafeHandle::null(),
        }
        .into();
        let raw_device = _cl_device_id::wrap(device);
        return DeviceKind::try_from_cl(raw_device).unwrap();
    }
}

impl DeviceImpl for Device {
    fn get_device_type(&self) -> cl_device_type {
        self.device_type
    }

    fn get_device_name(&self) -> String {
        self.name.clone()
    }

    fn is_available(&self) -> bool {
        true
    }
    fn get_platform(&self) -> WeakPtr<PlatformKind> {
        self.platform.clone()
    }

    fn create_queue(
        &self,
        _context: SharedPtr<ContextKind>,
        _device: SharedPtr<DeviceKind>,
    ) -> QueueKind {
        unimplemented!();
    }

    fn get_vendor_name(&self) -> String {
        unimplemented!()
    }

    fn get_vendor_id(&self) -> cl_uint {
        unimplemented!()
    }

    fn is_compiler_available(&self) -> bool {
        unimplemented!()
    }

    fn get_native_driver_version(&self) -> String {
        unimplemented!()
    }

    fn get_device_profile(&self) -> String {
        unimplemented!()
    }

    fn get_device_version_info(&self) -> String {
        unimplemented!()
    }

    fn get_extension_names(&self) -> &[&str] {
        unimplemented!()
    }

    fn get_extension_versions(&self) -> &[cl_version] {
        unimplemented!()
    }
}
