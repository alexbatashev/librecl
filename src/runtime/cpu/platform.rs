use lcl_derive::{self, ClObjImpl};

use super::Device;
use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::PlatformImpl;
use crate::interface::PlatformKind;
use crate::sync::{self, *};
use pytorch_cpuinfo::get_packages;
use std::ops::DerefMut;

#[derive(ClObjImpl)]
pub struct Platform {
    platform_name: String,
    devices: Vec<SharedPtr<DeviceKind>>,
    #[cl_handle]
    handle: UnsafeHandle<cl_platform_id>,
}

impl Platform {
    pub fn new() -> Platform {
        let platform_name = "LibreCL on Host CPU".to_owned();

        Platform {
            platform_name,
            devices: vec![],
            handle: UnsafeHandle::null(),
        }
    }

    pub fn create_platforms(platforms: &mut Vec<SharedPtr<PlatformKind>>) {
        let platform = Platform::new().into();
        let raw_platform = _cl_platform_id::wrap(platform);
        let mut cl_platform = PlatformKind::try_from_cl(raw_platform).unwrap();

        let packages = get_packages();

        for package in packages {
            let cl_device = Device::new(SharedPtr::downgrade(&cl_platform), package).into();
            cl_platform.deref_mut().add_device(cl_device);
        }

        platforms.push(cl_platform);
    }
}

impl PlatformImpl for Platform {
    fn get_platform_name(&self) -> &str {
        return &self.platform_name;
    }
    fn get_devices(&self) -> &[SharedPtr<DeviceKind>] {
        &self.devices
    }
    fn add_device(&mut self, device: SharedPtr<DeviceKind>) {
        self.devices.push(device);
    }
    fn create_context(
        &self,
        _devices: &[WeakPtr<DeviceKind>],
        _callback: cl_context_callback,
        _user_data: *mut libc::c_void,
    ) -> ContextKind {
        unimplemented!()
    }

    fn get_profile(&self) -> &str {
        return "FULL_PROFILE";
    }

    fn get_platform_version_info(&self) -> &str {
        unimplemented!()
    }

    fn get_extension_names(&self) -> &[&str] {
        unimplemented!()
    }
    fn get_extension_versions(&self) -> &[cl_version] {
        unimplemented!()
    }
    fn get_host_timer_resolution(&self) -> cl_ulong {
        unimplemented!()
    }
}
