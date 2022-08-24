use lcl_derive::{self, ClObjImpl};

use crate::api::cl_types::*;
use crate::interface::ContextKind;
use crate::interface::DeviceKind;
use crate::interface::PlatformImpl;
use crate::interface::PlatformKind;
use crate::sync::{self, *};

#[derive(ClObjImpl)]
pub struct Platform {
    #[cl_handle]
    handle: UnsafeHandle<cl_platform_id>,
}

impl Platform {
    pub fn default() -> Platform {
        Platform {
            handle: UnsafeHandle::null(),
        }
    }
    pub fn create_platforms(platforms: &mut Vec<SharedPtr<PlatformKind>>) {
        let platform = Platform::default().into();
        let raw_platform = _cl_platform_id::wrap(platform);
        let cl_platform = PlatformKind::try_from_cl(raw_platform).unwrap();
        platforms.push(cl_platform);
    }
}

impl PlatformImpl for Platform {
    fn get_platform_name(&self) -> &str {
        unimplemented!()
    }
    fn get_devices(&self) -> &[SharedPtr<DeviceKind>] {
        unimplemented!()
    }
    fn add_device(&mut self, _device: SharedPtr<DeviceKind>) {
        unimplemented!()
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
        unimplemented!()
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
