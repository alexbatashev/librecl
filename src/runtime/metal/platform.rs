use std::ops::DerefMut;

use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::PlatformImpl;
use crate::interface::PlatformKind;
use crate::sync::{self, *};
use lcl_derive::ClObjImpl;

use cpmetal::Device as MTLDevice;

use super::Context;
use super::Device;

#[derive(ClObjImpl)]
pub struct Platform {
    devices: Vec<SharedPtr<DeviceKind>>,
    name: String,
    extension_names: Vec<&'static str>,
    extension_versions: Vec<cl_version>,
    #[cl_handle]
    handle: UnsafeHandle<cl_platform_id>,
}

impl Platform {
    pub fn new() -> Platform {
        let extension_names = vec!["cl_khr_icd"];
        let extension_versions = vec![make_version(1, 0, 0)];
        let platform_name = std::format!("LibreCL Apple Metal Platform");
        return Platform {
            devices: vec![],
            name: platform_name,
            extension_names,
            extension_versions,
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

    fn get_profile(&self) -> &str {
        "FULL_PROFILE"
    }

    fn get_platform_version_info(&self) -> &str {
        // TODO find way to identfy version
        "over Apple Metal"
    }

    fn get_platform_name(&self) -> &str {
        self.name.as_str()
    }

    fn get_extension_names(&self) -> &[&str] {
        &self.extension_names
    }

    fn get_extension_versions(&self) -> &[cl_version] {
        &self.extension_versions
    }

    fn get_host_timer_resolution(&self) -> cl_ulong {
        0 // TODO return actual value
    }
}
