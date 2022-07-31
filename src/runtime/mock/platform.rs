use ocl_type_wrapper::{self, ClObjImpl};

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
        unimplemented!();
    }
    fn get_devices(&self) -> &[SharedPtr<DeviceKind>] {
        unimplemented!();
    }
    fn add_device(&mut self, device: SharedPtr<DeviceKind>) {
        unimplemented!();
    }
    fn create_context(
        &self,
        devices: &[WeakPtr<DeviceKind>],
        callback: cl_context_callback,
        user_data: *mut libc::c_void,
    ) -> ContextKind {
        unimplemented!();
    }
}
