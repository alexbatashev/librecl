use std::ops::DerefMut;

use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::PlatformImpl;
use crate::interface::PlatformKind;
use crate::sync::{self, *};
use ocl_type_wrapper::ClObjImpl;

use metal_api::Device as MTLDevice;

use super::Context;
use super::Device;

#[derive(ClObjImpl)]
pub struct Platform {
    devices: Vec<SharedPtr<DeviceKind>>,
    name: String,
    #[cl_handle]
    handle: UnsafeHandle<cl_platform_id>,
}

impl Platform {
    pub fn new() -> Platform {
        let platform_name = std::format!("LibreCL Apple Metal Platform");
        return Platform {
            devices: vec![],
            name: platform_name,
            handle: UnsafeHandle::null(),
        };
    }

    pub fn create_platforms(platforms: &mut Vec<SharedPtr<PlatformKind>>) {
        let native_platform = Platform::new().into();
        let raw_platform = _cl_platform_id::wrap(native_platform);
        let mut platform: SharedPtr<PlatformKind> =
            PlatformKind::try_from_cl(raw_platform).unwrap();

        let all_devices = MTLDevice::all();

        for d in all_devices {
            let device = Device::new(&platform, d);
            platform.deref_mut().add_device(device);
        }

        platforms.push(platform);
    }
}

impl PlatformImpl for Platform {
    fn get_platform_name(&self) -> &str {
        return self.name.as_str();
    }

    fn get_devices(&self) -> &[SharedPtr<DeviceKind>] {
        return &self.devices.as_slice();
    }

    fn add_device(&mut self, device: SharedPtr<DeviceKind>) {
        self.devices.push(device);
    }

    fn create_context(
        &self,
        devices: &[WeakPtr<DeviceKind>],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> ContextKind {
        return Context::new(devices, callback, user_data);
    }
}
