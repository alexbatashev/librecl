use crate::api::cl_types::ClObjectImpl;
use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceImpl;
use crate::interface::DeviceKind;
use crate::interface::PlatformKind;
use crate::interface::QueueKind;
use crate::sync::UnsafeHandle;
use crate::sync::{self, *};
use ocl_type_wrapper::ClObjImpl;
use pytorch_cpuinfo::Package;

#[derive(ClObjImpl)]
pub struct Device {
    device_type: cl_device_type,
    name: String,
    platform: WeakPtr<PlatformKind>,
    #[cl_handle]
    handle: UnsafeHandle<cl_device_id>,
}

impl Device {
    pub fn new(platform: WeakPtr<PlatformKind>, package: &Package) -> SharedPtr<DeviceKind> {
        let device = Device {
            device_type: cl_device_type::CPU,
            name: package.name.clone(),
            platform,
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
}
